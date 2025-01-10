import cv2
import numpy as np

from Utils.converter import *
from scipy.optimize import differential_evolution
import pandas as pd
import copy
import os

import colorsys
import time

from Utils.generalCV import overlap_image_with_label
from Utils.ultrasoundPreprocessingFunctions import compute_phase_symmetry
from multiprocessing import Pool
from skimage import exposure
np.set_printoptions(precision=3)
import sys
from Utils.generalCV import *







def intensity_to_rgb_vectorized(intensities, min_intensity=0, max_intensity=1):
    intensities[intensities>1]=1
    normalized_intensities = (intensities - min_intensity) / (max_intensity - min_intensity)
    hues = (1.0 - normalized_intensities) * 0.666
    saturations = np.ones_like(hues)
    values = np.ones_like(hues)
    rgb = np.stack([colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hues, saturations, values)], axis=0)
    return np.clip(rgb, 0, 1)


def image2PCD(img, calibration_t, calibration_euler, scale_X, scale_Y,down_sample_stride=2):
    height, width = img.shape[:2]
    T_scale = np.eye(4)
    T_scale[0, 0] = scale_X
    T_scale[1, 1] = scale_Y


    if not down_sample_stride is None:
        img[::down_sample_stride, :] = 0
        img[:, ::down_sample_stride] = 0
    rows_index, cols_index = np.where(img > 0)
    intensities = img[rows_index, cols_index]

    xyz_pixels = np.stack([cols_index+0.5, rows_index+0.5, np.zeros_like(rows_index), np.ones_like(rows_index)], axis=1).T
    calibrationT = vectorToMatrix(calibration_t, calibration_euler)  # Assuming vectorToMatrix is defined
    xyz_world = (calibrationT @ T_scale @ xyz_pixels)[:3, :].T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_world)

    if len(pcd.points)>0:
        colors = intensity_to_rgb_vectorized(intensities)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, intensities



def enhanceContrast(img, ct_file_paths, calibration_t, calibration_euler, scale_X, scale_Y, T_tracking,
                    intcDisThreshold):
    height, width = img.shape[:2]
    T_scale = np.eye(4)
    T_scale[0, 0] = scale_X
    T_scale[1, 1] = scale_Y

    rows_index, cols_index = np.where(img >= 0)

    xyz_pixels = np.stack([cols_index + 0.5, rows_index + 0.5, np.zeros_like(rows_index), np.ones_like(rows_index)],
                          axis=1).T
    calibrationT = vectorToMatrix(calibration_t, calibration_euler)  # Assuming vectorToMatrix is defined
    xyz_world = (T_tracking @ calibrationT @ T_scale @ xyz_pixels)[:3, :].T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_world)
    pcd_CT = o3d.geometry.PointCloud()
    for ct_file_path in ct_file_paths:

        mesh_single = o3d.io.read_triangle_mesh(ct_file_path)
        pcd_single = mesh_single.sample_points_uniformly(50000)
        # o3d.visualization.draw_geometries([pcd_single,pcd])
        # print(f"finish sampling points from one CT")
        dists = pcd.compute_point_cloud_distance(pcd_single)

        dists = np.asarray(dists)
        if dists.min() < 3:
            ind_seperate = np.where(dists < 3)[0]

            if img[rows_index[ind_seperate], cols_index[ind_seperate]].max() > 150:
                a = img[rows_index[ind_seperate], cols_index[ind_seperate]]
                if a[a > np.percentile(a, 95)].mean() / 255. < 0.4:
                    continue
                # print(f"{ct_file_path}: {a[a > np.percentile(a, 95)].mean() / 255.}")
                pcd_single = mesh_single.sample_points_uniformly(1000000)
                pcd_CT += pcd_single

        # pcd_CT += pcd_single

    dists = pcd.compute_point_cloud_distance(pcd_CT)
    dists = np.asarray(dists)
    ind = np.where(dists > intcDisThreshold)[0]
    img_intersected = copy.deepcopy(img)
    img_intersected[rows_index[ind], cols_index[ind]] = 0
    img_intersected[img_intersected < 10] = 0
    return img_intersected, pcd_CT


def selection_based_on_distance(source, target, thresh=5):
    dists=source.compute_point_cloud_distance(target)
    dists=np.asarray(dists)
    ind=np.where(dists<thresh)[0]

    if ind.size>0:
        # print(f"mean distance without outliers: {dists[ind].mean()}")
        return source.select_by_index(ind)
    else:
        return source



def objective_function(params,pcd_US,pcd_CT,intensity_a,threshold,initial_translation,initial_euler):
    #copy a pcd, in case overwrite in parallel execution
    pcd_US=copy.deepcopy(pcd_US)
    translation = params[:3]
    rotation_params = params[3:]  # Assuming an Euler angle parameterization
    rotation_matrix = R.from_euler('xyz', rotation_params, degrees=True).as_matrix()

    # Construct the transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    pcd_US.transform(T)
    dists = pcd_US.compute_point_cloud_distance(pcd_CT)
    dists = np.asarray(dists)
    ind = np.where(dists < threshold)[0]
    if len(ind)==0:
        return 0
    mean_method=False
    if mean_method:
        score_exp = -(np.mean((1 * (intensity_a[ind]))))
        # print(f"ratio:{(np.sum(np.exp(5*(intensity_a[ind]))))/(np.mean(np.exp(5*(intensity_a[ind]))))}")

        distance_optimized = compute_fiducials_deviation(initial_translation, initial_euler, translation,
                                                         rotation_params)
        regularization = 0.1 * distance_optimized

        return score_exp + regularization
    else:
        score_exp = -(np.sum(np.exp(5*(intensity_a[ind]))))
        # print(f"ratio:{(np.sum(np.exp(5*(intensity_a[ind]))))/(np.mean(np.exp(5*(intensity_a[ind]))))}")

        distance_optimized = compute_fiducials_deviation(initial_translation, initial_euler, translation,
                                                         rotation_params)
        regularization = 3000 * distance_optimized

        return score_exp + regularization




def compute_fiducials_deviation(initial_translation,initial_euler,translation,euler):
    fiducials = np.asarray([[-30.3539, -25.101, -0.4395],
                            [25.7953, -25.101, -0.5479],
                            [37.0174, -25.101, -28.2339],
                            [-30.6348, -25.101, -57.3485]])
    distance_moved=0
    for fiducial in fiducials:
        rotation_matrix = R.from_euler('xyz', euler, degrees=True).as_matrix()
        fiducial_transformed=rotation_matrix@fiducial+translation
        rotation_matrix_initial=R.from_euler('xyz', initial_euler, degrees=True).as_matrix()
        fiducial_initial=rotation_matrix_initial@fiducial+initial_translation
        distance_moved+=np.linalg.norm(fiducial_initial-fiducial_transformed)
    return  distance_moved/4.


def initialize_population_uniformly(num_particles, lb, ub):
    dimensions = len(lb)
    population = np.random.rand(num_particles, dimensions) * (np.array(ub) - np.array(lb)) + np.array(lb)
    return population

def generateLabel(img, calibration_t, calibration_euler, scale_X, scale_Y, T_tracking,CT_pcd_lowRes,CT_pcd_highRes,
                  intensity_threhold=-1,optimized_tracking=False):
    height, width = img.shape[:2]
    T_scale = np.eye(4)
    T_scale[0, 0] = scale_X
    T_scale[1, 1] = scale_Y
    # img = cv2.GaussianBlur(img, (7, 7), sigmaX=3,sigmaY=3)

    rows_index, cols_index = np.where(img > intensity_threhold)
    intensities = img[rows_index, cols_index]

    xyz_pixels = np.stack([cols_index+0.5, rows_index+0.5, np.zeros_like(rows_index), np.ones_like(rows_index)], axis=1).T
    calibrationT = vectorToMatrix(calibration_t, calibration_euler)  # Assuming vectorToMatrix is defined
    xyz_world = (T_tracking @ calibrationT @ T_scale @ xyz_pixels)[:3, :].T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_world)

    # colors = self.intensity_to_rgb_vectorized(intensities)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_CT_intersected = o3d.geometry.PointCloud()
    idx_CT_pcd_highRes=[]
    for i,pcd_lowRes in enumerate(CT_pcd_lowRes):
        dists = pcd.compute_point_cloud_distance(pcd_lowRes)
        dists=np.asarray(dists)
        if dists.min() < 3:
            if optimized_tracking:
                ind_seperate = np.where(dists < 0.5)[0]
                pixels_intersected = img[rows_index[ind_seperate], cols_index[ind_seperate]]
                threshold_value=30/6
                # threshold_value = 30
                if len(pixels_intersected[pixels_intersected > threshold_value])==0:
                    continue
                if pixels_intersected[pixels_intersected > threshold_value].mean() / 255.>0.4/3:
                    # if pixels_intersected[pixels_intersected > threshold_value].mean() / 255. > 0.4 :
                    # print(pixels_intersected[pixels_intersected > threshold_value].mean() / 255.)
                    # pcd_CT_intersected += self.CT_pcd_highRes[i]
                    idx_CT_pcd_highRes+=[i]
            else:
                ind_seperate = np.where(dists < 3)[0]
                pcd_CT_intersected += CT_pcd_highRes[i]
                idx_CT_pcd_highRes += [i]
            if len(ind_seperate)==0:
                continue

            # if pixels_intersected[pixels_intersected > np.percentile(pixels_intersected, 95)].mean() / 255.>0.75:
            #     pcd_CT_intersected += self.CT_pcd_highRes[i]
                # print(f": {pixels_intersected[pixels_intersected > np.percentile(pixels_intersected, 90)].mean()/255.}")
        # else:
        # print('no')
    if len(idx_CT_pcd_highRes)==0:
        return None,None,None

    Label_full = np.zeros_like(img)
    for j,idx in enumerate(idx_CT_pcd_highRes):
        dists = pcd.compute_point_cloud_distance(CT_pcd_highRes[idx])
        dists = np.asarray(dists)
    # return None, None, None
        distance_threshold = 0.1
        ind = np.where(dists < distance_threshold)[0]
        Label_full[rows_index[ind], cols_index[ind]] +=((j+1)*(j+1))
    # mask is full CT projection, mask_selected is the part that is only visibilt in US
    # intensities_selected[intensities_selected<10]=0
    Label_full[Label_full>0]=255
    return pcd, intensities, Label_full

def process_rows(args):
    visualization_flag = False
    dataFolder, index_range, row_indices = args
    calibration_t = [26.44694442, -0.52572229, 128.00100047]
    calibration_euler = [92.48865621, -0.46874914, 179.2277322]
    scale_X = 0.05392
    scale_Y = 0.05392
    # Load point cloud and precompute any static resources as before
    # All previously setup code for CT data loading and processing goes here

    ct_file_folder = os.path.join(dataFolder, 'CT_models')
    ct_file_paths = []
    # pcd_CT=o3d.geometry.PointCloud()
    CT_pcd_lowRes = []
    CT_pcd_highRes = []
    for ct_file_name in os.listdir(ct_file_folder):
        ct_file_path = os.path.join(ct_file_folder, ct_file_name)
        mesh_single = o3d.io.read_triangle_mesh(ct_file_path)
        CT_pcd_lowRes.append(mesh_single.sample_points_uniformly(50000))
        CT_pcd_highRes.append(mesh_single.sample_points_uniformly(10000000))
    for ct_file_name in os.listdir(ct_file_folder):
        ct_file_paths.append(os.path.join(ct_file_folder, ct_file_name))
        # mesh_single = o3d.io.read_triangle_mesh(os.path.join(ct_file_folder, ct_file_name))
        # pcd_single = mesh_single.sample_points_uniformly(100000)
        # pcd_CT += pcd_single
    # o3d.visualization.draw_geometries([pcd_CT])

    # Prepare the DataFrame
    sweep_df = pd.read_csv(os.path.join(dataFolder, 'sweep_initial.csv'))
    sweep_df = sweep_df.loc[row_indices]
    sweep_df['x_optimized'] = np.nan

    img_folder = os.path.join(dataFolder, 'UltrasoundImages')
    row_index_incide_core = 0
    for i, row in sweep_df.iterrows():
        row_index_incide_core += 1
        if row_index_incide_core % 10 == 0:
            print(
                f"Core :{index_range}, Processing row: {row_index_incide_core}/{len(sweep_df)}, timestamp: {row['timestamp']}")
        else:
            continue
        # print(f"Processing row: {i}/{index_range}")
        # if row['timestamp'] not in [760909,801844,813710]:
        # if row['timestamp'] not in [ 908085]:
        #     continue
        start = time.time()
        img_path = os.path.join(img_folder, f"{int(row['timestamp'])}.png")

        if os.path.isfile(img_path):

            # Load and process each image as per the original code
            T_US_to_CT_initial = vectorToMatrix([row['x'], row['y'], row['z']],
                                        [row['euler_x'], row['euler_y'], row['euler_z']])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # img=apply_histogram_equalization(img)
            # img_ps = compute_phase_symmetry(img, num_scales=2)
            # img_normalized = normalize_grayscale_image_dynamic_cv2(img)
            intcDisThreshold = 2.5

            img_enhanced, pcd_CT = enhanceContrast(img, ct_file_paths,
                                                   calibration_t, calibration_euler, scale_X, scale_Y, T_US_to_CT_initial,
                                                   intcDisThreshold)

            if visualization_flag:
                cv2.imshow('original', img)
                # cv2.imshow('equalized',img_equalized)
                # cv2.imshow('ps', img_ps)
                cv2.imshow('enhanced', img_enhanced)
                # cv2.waitKey(0)

            num_points = len(pcd_CT.points)

            if num_points == 0:
                continue

            # Set the color of all points to blue
            blue_color = [0, 0, 1]  # RGB values for blue
            pcd_CT.colors = o3d.utility.Vector3dVector([blue_color] * num_points)

            img_enhanced = img_enhanced / 255.0

            # img_enhanced=copy.deepcopy(img_normalized)
            # img_enhanced[img_enhanced>0.8]=0.8
            # img_enhanced[img_enhanced < 0.05] = 0

            # a = copy.deepcopy(img_enhanced)
            # b = copy.deepcopy(img_enhanced)
            # a[img_ps > 0.05] *= 2
            # b[np.logical_and(img_ps > 0.05, b > 0)] = 1
            # img_enhanced = np.minimum(a, b)

            if visualization_flag:
                cv2.imshow('original', img)
                # cv2.imshow('equalized',img_equalized)
                # cv2.imshow('ps', img_ps)
                cv2.imshow('enhanced', img_enhanced)
                cv2.imshow("image2pcd", img_enhanced)
                # cv2.waitKey(0)
            pcd_US_selected, intensities = image2PCD(img_enhanced, calibration_t, calibration_euler, scale_X, scale_Y,
                                                     down_sample_stride=2)
            if len(intensities)==0:
                continue
            mean_intensity_gt_90 = intensities[intensities > np.percentile(intensities, 95)].mean()
            # print(f"mean intensity > 95%: {mean_intensity_gt_90}")
            if mean_intensity_gt_90 < 0.45:
                continue
            # if np.percentile(intensities,100)<0.75:
            #     continue
            initial_translation = [row[k] for k in ['x', 'y', 'z']]
            init_eulers = [row[k] for k in ['euler_x', 'euler_y', 'euler_z']]

            pcd_US_selected_initial_registered = copy.deepcopy(pcd_US_selected).transform(T_US_to_CT_initial)

            dists = pcd_US_selected_initial_registered.compute_point_cloud_distance(pcd_CT)
            dists = np.asarray(dists)
            if len(pcd_US_selected.points) == 0 or dists.min() > 0.9:
                # sweep_df.loc[i, ['x_optimized', 'y_optimized', 'z_optimized']] = initial_translation
                # sweep_df.loc[i, ['euler_x_optimized', 'euler_y_optimized', 'euler_z_optimized']] = init_eulers
                # print(f"used time: {time.time() - start}, img: {img_path}")
                continue
            range = 1
            range_angle = 1
            bounds = np.asarray([(row['x'] - range, row['x'] + range),
                                 (row['y'] - range, row['y'] + range),
                                 (row['z'] - range, row['z'] + range),
                                 (row['euler_x'] - range_angle, row['euler_x'] + range_angle),
                                 (row['euler_y'] - range_angle, row['euler_y'] + range_angle),
                                 (row['euler_z'] - range_angle, row['euler_z'] + range_angle),
                                 ])
            distance_threshold = 0.5  # kepp points with distance to CT <0.5
            pcd_CT_selected = selection_based_on_distance(pcd_CT, pcd_US_selected_initial_registered, 2)
            # pcd_CT_selected.points=pcd_CT_selected.points[::2]
            # pcd_US_selected=pcd_US_selected.voxel_down_sample(0.1)

            init_pop = initialize_population_uniformly(200, bounds[:, 0], bounds[:, 1])

            result = differential_evolution(objective_function, bounds,
                                            args=(pcd_US_selected, pcd_CT_selected, intensities, distance_threshold,
                                                  initial_translation, init_eulers),
                                            # maxiter=3,popsize=100,
                                            maxiter=30,
                                            workers=1, polish=True,
                                            tol=0.05,
                                            # atol=500,

                                            # mutation=0.9,
                                            # init='sobol',
                                            # init=inist_pop,
                                            disp=True if visualization_flag else False,
                                            seed=0,  # recombination=0.9
                                            # callback=callback,
                                            # strategy='rand2bin',
                                            # callback=MinimizeStopper()
                                            # x0=initial_translation + init_eulers,
                                            # #updating='deferred'#,strategy='rand2bin',init='sobol',#
                                            # init=init_pop,
                                            # updating='deferred'
                                            )
            used_time = time.time() - start
            T_optimized = vectorToMatrix(result.x[:3], result.x[3:])
            distance_optimized = np.linalg.norm(np.asarray(result.x[:3]) - np.asarray(initial_translation))
            angle_optimized = np.linalg.norm(np.asarray(result.x[3:]) - np.asarray(init_eulers))
            distance_optimized_fiducials = compute_fiducials_deviation(initial_translation, init_eulers, result.x[:3],
                                                                       result.x[3:])
            print(
                f"used time: {used_time}, distance_optimized_fiducials {distance_optimized_fiducials}, distanced_optimized: {distance_optimized}, optimization result: {result.fun}, img: {img_path}")
            if visualization_flag:
                print(f"pcd_CT has size: {len(pcd_CT_selected.points)}")
                print(
                    f"optimized translation: {result.x[:3] - initial_translation}, rotations: {result.x[3:] - init_eulers}")

                print(f"current i: {i}, min dis: {dists.min()}")
                # o3d.visualization.draw_geometries([pcd_US_selected])
                # o3d.visualization.draw_geometries([pcd_US_rough_registered, pcd_CT])
                o3d.visualization.draw_geometries(
                    [copy.deepcopy(pcd_US_selected).transform(T_US_to_CT_initial), pcd_CT_selected])
                o3d.visualization.draw_geometries([copy.deepcopy(pcd_US_selected).transform(T_optimized), pcd_CT])

                pcd_US_selected_transformed = copy.deepcopy(pcd_US_selected).transform(T_optimized)
                dists = np.asarray(pcd_US_selected_transformed.compute_point_cloud_distance(pcd_CT_selected))
                ind = np.where(dists <= 2)[0]
                pcd_US_selected_transformed = pcd_US_selected_transformed.select_by_index(ind)
                print(dists[ind].mean())
                o3d.visualization.draw_geometries([pcd_US_selected_transformed, pcd_CT_selected])


                # o3d.visualization.draw_geometries([copy.deepcopy(pcd_US_selected).transform(T_tracking), pcd_CT_selected])


            sweep_df.loc[i, ['x_optimized', 'y_optimized', 'z_optimized']] = result.x[:3]
            sweep_df.loc[i, ['euler_x_optimized', 'euler_y_optimized', 'euler_z_optimized']] = result.x[3:]
            sweep_df.loc[i, ['distance_optimized']] = distance_optimized
            sweep_df.loc[i, ['distance_optimized_fiducials']] = distance_optimized_fiducials
            sweep_df.loc[i, ['angle_optimized']] = angle_optimized
            sweep_df.loc[i, ['x_diff', 'y_diff', 'z_diff']] = np.abs(result.x[:3] - initial_translation)
            sweep_df.loc[i, ['euler_x_diff', 'euler_y_diff', 'euler_z_diff']] = np.abs(result.x[3:] - init_eulers)

            sweep_df.loc[i, ['used_time']] = used_time

            T_US_to_CT_optimized = vectorToMatrix(result.x[:3],result.x[3:])
            _, _, label_full = generateLabel(img, calibration_t,
                                                                       calibration_euler,
                                                                       scale_X, scale_Y,
                                                                       T_US_to_CT_initial, CT_pcd_lowRes,
                                                                       CT_pcd_highRes,
                                                                       intensity_threhold=-1)
            _, _, label_full_optimized = generateLabel(img, calibration_t,
                                                                       calibration_euler,
                                                                       scale_X, scale_Y,
                                                                       T_US_to_CT_optimized, CT_pcd_lowRes,
                                                                       CT_pcd_highRes,
                                                                       intensity_threhold=-1)
            result_image_inital=overlap_image_with_label(img,label_full)
            result_image_optimized=overlap_image_with_label(img,label_full_optimized)
            result_image_merged=merge_images_horizontally([result_image_inital,result_image_optimized])
            # cv2.imshow("result",result_image_merged)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(dataFolder,"optimization_results",f"{int(row['timestamp'])}_res.png"),result_image_merged)


            # Continue processing as before

    # Save partial results or return processed data
    sweep_df = sweep_df[sweep_df['x_optimized'].notna()]
    processed_path = os.path.join(dataFolder, f'processed_part_{index_range}.csv')
    sweep_df.to_csv(processed_path, index=False)

    return processed_path

def parallel_process_dataframe(dataFolder, num_processes):
    if not os.path.isdir(dataFolder):
        return
    sweep_df = pd.read_csv(os.path.join(dataFolder, 'sweep_initial.csv'))[:]
    row_splits = np.array_split(sweep_df.index, num_processes)

    args = [(dataFolder, i, row_splits[i]) for i in range(num_processes)]
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_rows, args)

    # Optionally, combine results back into a single DataFrame
    final_df = pd.DataFrame()
    for result in results:
        if result =="-1":
            continue
        part_df = pd.read_csv(result)
        final_df = pd.concat([final_df, part_df], ignore_index=True)
        os.remove(result)
    final_df.to_csv(os.path.join(dataFolder, 'sweepProcessedAndOptimized.csv'), index=False)



import multiprocessing

if __name__=='__main__':
    dataFolder="./record_folder"
    parallel_process_dataframe(dataFolder, num_processes=1)
