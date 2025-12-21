import os
import time
import numpy as np
import pandas as pd
from skimage.io import imread

from Utility.converter import *
from multiprocessing import Pool
from functools import partial
import argparse

def downsample_to_n_random(pcd: o3d.geometry.PointCloud, target_n: int, seed: int = 42) -> o3d.geometry.PointCloud:
    """
    Return a point cloud with exactly target_n points by random sampling (no replacement).
    If pcd has fewer than target_n points, it returns a copy unchanged.
    Preserves colors and normals if present.
    """
    if target_n <= 0:
        raise ValueError("target_n must be > 0")

    n = len(pcd.points)
    if n <= target_n:
        return pcd.clone() if hasattr(pcd, "clone") else o3d.geometry.PointCloud(pcd)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=target_n, replace=False)

    pts = np.asarray(pcd.points)[idx]
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)

    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx])
    if pcd.has_normals():
        out.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[idx])

    return out

def construct_pcd_from_df(df_chunk, use_optimized_tracking: bool = False):
    df_chunk = df_chunk.reset_index(drop=True)
    label_pcd_merged = []

    for index, row in df_chunk.iterrows():
        label = imread(row['label_path'], as_gray=True)
        if 255 not in np.unique(label):
            continue

        # Choose which tracking data to use
        if use_optimized_tracking:
            T_tracking = vectorToMatrix([row['x_optimized'], row['y_optimized'], row['z_optimized']],
                                        [row['euler_x_optimized'], row['euler_y_optimized'], row['euler_z_optimized']])
        else:
            T_tracking = vectorToMatrix([row['x'], row['y'], row['z']],
                                        [row['euler_x'], row['euler_y'], row['euler_z']])

        # the coordinate system of the ultrasound image is defined at the top-left corner
        T_scale = np.eye(4)
        T_scale[0, 0] = row['scale_X']
        T_scale[1, 1] = row['scale_Y']

        # keep only bone pixels
        rows_index, cols_index = np.where(abs(label) > 0)
        if len(rows_index) < 2:
            return None

        xyz_pixels = np.stack(
            [cols_index + 0.5, rows_index + 0.5, np.zeros_like(rows_index), np.ones_like(rows_index)],
            axis=1
        ).T  # homogenous coordinates with [x,y,z,1]. z is always 0 for the image plane

        calibrationT = vectorToMatrix(row['calibration_t'], row['calibration_euler'])

        T_img_2_world = T_tracking @ calibrationT @ T_scale
        xyz_world = (T_img_2_world @ xyz_pixels)

        label_pcd_merged.append(xyz_world.T)

    if label_pcd_merged:
        return np.concatenate(label_pcd_merged)
    return None

def filter_out_untargeted_anatomy_points(US_pcd, bone_segmentation_pcds):
    nearest_dists = np.ones(len(US_pcd.points)) * 10000
    idx_final = np.ones(len(US_pcd.points)) * 5
    for anatomy_idx, anatomy_pcd_gt in enumerate(bone_segmentation_pcds):
        dist = np.asarray(US_pcd.compute_point_cloud_distance(anatomy_pcd_gt))
        idx_temp = nearest_dists > dist
        nearest_dists[idx_temp] = dist[idx_temp]
        idx_final[idx_temp] = anatomy_idx
    vals, counts = np.unique(idx_final, return_counts=True)
    most_freq_val = vals[np.argmax(counts)]
    indices = np.where(idx_final == most_freq_val)[0]
    return US_pcd.select_by_index(indices)

def main_3D_reconstruction(dataset_root_folder,use_optimized_pose=False ,use_pred_label=True):

    # calibration results. Note that the special details are the same for all sweeps.
    calibration_t = [26.44694442, -0.52572229, 128.00100047]  # unit: mm
    calibration_euler = [92.48865621, -0.46874914, 179.2277322]  # euler angles in degrees
    scale_X = 0.05392  # mm/pixel
    scale_Y = 0.05392  # mm/pixel


    for specimen_id in range(1, 15):
        anatomies=["fibula", "tibia", "foot"]
        specimen_folder = os.path.join(dataset_root_folder, f"specimen{specimen_id:02d}")
        UltrasoundRecords_folder = os.path.join(specimen_folder, "ultrasound_records")

        # read CT model data


        bone_segmentation_meshes=[o3d.io.read_triangle_mesh(os.path.join(specimen_folder, "CT_bone_segmentations", f"{anatomy}.stl")) for anatomy in anatomies]
        bone_segmentation_pcds=[mesh.sample_points_uniformly(500000) for mesh in bone_segmentation_meshes]
        bone_segmentation_meshes_merged = o3d.io.read_triangle_mesh(
            os.path.join(specimen_folder, "CT_bone_segmentations", "CT_bone_model_merged.stl"))
        bone_segmentation_pcds_merged = bone_segmentation_meshes_merged.sample_points_uniformly(1000000)

        for anatomy in anatomies:
            anatomy_folder = os.path.join(UltrasoundRecords_folder, anatomy)
            for record_folder_name in os.listdir(anatomy_folder):
                record_folder = os.path.join(anatomy_folder, record_folder_name)
                if not os.path.isdir(record_folder):
                    print(f"record folder does not exist: {record_folder}")
                    continue
                tracking_df = pd.read_csv(os.path.join(record_folder, 'tracking.csv'))

                if use_pred_label==True:
                    label_folder = os.path.join(record_folder, 'Labels_pred')
                    timestamps_target = [
                        int(file.split('_')[0]) for file in os.listdir(label_folder)
                        if file.endswith("_label_pred.png")
                    ]
                    tracking_df['label_path'] = tracking_df['timestamp'].apply(
                        lambda x: os.path.join(label_folder, f"{x}_label_pred.png")
                    )

                    reconstruction_folder = os.path.join(record_folder, "3D_reconstructions","with_pred_labels")
                    os.makedirs(reconstruction_folder, exist_ok=True)
                else:
                    label_folder = os.path.join(record_folder, 'Labels')
                    timestamps_target = [
                        int(file.split('_')[0]) for file in os.listdir(label_folder)
                        if file.endswith("_label.png")
                    ]
                    tracking_df['label_path'] = tracking_df['timestamp'].apply(
                        lambda x: os.path.join(label_folder, f"{x}_label.png")
                    )

                    reconstruction_folder = os.path.join(record_folder, "3D_reconstructions","with_GT_labels")
                    os.makedirs(reconstruction_folder, exist_ok=True)


                tracking_df = tracking_df[tracking_df['timestamp'].isin(timestamps_target)]

                # update the details for each frame

                tracking_df['calibration_t'] = [calibration_t] * len(tracking_df)
                tracking_df['calibration_euler'] = [calibration_euler] * len(tracking_df)
                tracking_df['scale_X'] = [scale_X] * len(tracking_df)
                tracking_df['scale_Y'] = [scale_Y] * len(tracking_df)

                # Chunking the DataFrame
                num_processes = 6
                pool = Pool(processes=num_processes)
                chunk_size = int(np.ceil(len(tracking_df) / num_processes))
                df_chunks = [tracking_df.iloc[i:i + chunk_size] for i in range(0, len(tracking_df), chunk_size)]

                # Processing in parallel (pass the flag into each worker)
                worker = partial(construct_pcd_from_df, use_optimized_tracking=use_optimized_pose)
                results = pool.map(worker, df_chunks)
                pool.close()
                pool.join()

                # Drop Nones (in case any chunk returned None)
                results = [r for r in results if r is not None]
                if not results:
                    print("No points reconstructed for this record (all chunks empty).")
                    continue

                # Combining results
                xyz_with_feature = np.concatenate(results)
                xyz = xyz_with_feature[:, :3]
                reconstruction_pcd = o3d.geometry.PointCloud()
                reconstruction_pcd.points = o3d.utility.Vector3dVector(xyz)
                # reconstruction_pcd=reconstruction_pcd.farthest_point_down_sample(40000)
                # reconstruction_pcd=reconstruction_pcd.uniform_down_sample(0.1)
                reconstruction_pcd=downsample_to_n_random(reconstruction_pcd,int(1e6))

                # Each record targets a single anatomy. However, surfaces from nearby bones can also appear in the scan.
                # For example, during a fibula sweep, parts of the tibia may be captured in regions where the bones are close.
                # The filtered point cloud below removes points likely belonging to non-target (untargeted) anatomies.
                # This can be useful when you want a reconstruction focused only on the intended anatomy.
                reconstruction_pcd_filtered=filter_out_untargeted_anatomy_points(reconstruction_pcd,bone_segmentation_pcds)

                dis = np.asarray(reconstruction_pcd.compute_point_cloud_distance(bone_segmentation_pcds_merged))
                print(f"{record_folder} CD distance from US-reconstruction to CT-pcd: {dis.mean()}")
                if use_optimized_pose:
                    o3d.io.write_point_cloud(
                        os.path.join(reconstruction_folder, "reconstruction_pcd_optimizedPose.xyz"),
                        reconstruction_pcd
                    )
                    o3d.io.write_point_cloud(
                        os.path.join(reconstruction_folder, "reconstruction_pcd_filtered_optimizedPose.xyz"),
                        reconstruction_pcd_filtered
                    )
                else:
                    o3d.io.write_point_cloud(
                        os.path.join(reconstruction_folder, "reconstruction_pcd.xyz"),
                        reconstruction_pcd
                    )

                    o3d.io.write_point_cloud(
                        os.path.join(reconstruction_folder, "reconstruction_pcd_filtered.xyz"),
                        reconstruction_pcd_filtered
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process ultrasound images.")

    parser.add_argument('--dataset_root_folder', type=str, default="./data/AI_Ultrasound_dataset",
                        help='Root directory for the dataset')
    parser.add_argument('--use_optimized_pose', type=bool, default=False,
                        help='use optimized pose or not')
    parser.add_argument('--use_pred_label', type=bool, default=True,
                        help='use label predictions or GT')
    args = parser.parse_args()
    main_3D_reconstruction(dataset_root_folder=args.dataset_root_folder, use_optimized_pose=args.use_optimized_pose,use_pred_label=args.use_pred_label)
