import numpy as np

import cv2
import open3d as o3d
from tqdm import  tqdm
from scipy.spatial.transform import Rotation as R

def vectorToMatrix(t, rotation_vector, quat=False):
       T=np.eye(4)
       if quat:
              r=R.from_quat(rotation_vector)
       else:

              r=R.from_euler('xyz', rotation_vector, degrees=True)
       T[:3,:3]=r.as_matrix()
       T[:3,3]=np.asarray(t)
       return T

def MatrixToVector(T):
       r=R.from_matrix(T[:3,:3]).as_euler('xyz',degrees=True)
       return np.array([T[0,3],T[1,3],T[2,3],r[0],r[1],r[2]])


def vectorToMatrixInv(t,euler_xyz):
       T=np.eye(4)
       r=R.from_euler('xyz',euler_xyz,degrees=True)
       T[:3,:3]=r.as_matrix().T
       T[:3,3]=-np.asarray(t)
       return T

def sweepToPCD(sweep,temporalOff,calibrationT):
       pcd=o3d.geometry.PointCloud()
       sweep.tracking().temporalOffset=temporalOff
       sweep.tracking().calibration=calibrationT
       sweep_np=sweep.numpy()
       for i in range(sweep_np.shape[0]):
              img_imf=sweep.img(i)
              img_np=np.array((sweep_np[i]))
              rows_index, cols_index,_ = np.where(img_np > 0)
              num_points = len(cols_index)
              xyz_pixels = np.stack([cols_index, rows_index, np.zeros(num_points), np.ones(num_points)], axis=0)
              xyz_img = np.dot(img_imf.pixelToImageMatrix(), xyz_pixels)
              xyz_world = np.dot(sweep.matrix(i), xyz_img)
              xyz_world = xyz_world[:3, :]
              xyz_world = xyz_world.T
              pcd.points.extend(xyz_world)
       pcd = pcd.voxel_down_sample(voxel_size=0.1)
       return pcd


def transform_Nx3_array(T,xyz):
       xyz_homo=np.hstack([xyz,np.ones((xyz.shape[0],1))])
       return ((T@xyz_homo.T).T)[:,:3]


def array2open3dPCD(xyz):
       pcd=o3d.geometry.PointCloud()
       pcd.points=o3d.utility.Vector3dVector(xyz)
       pcd.paint_uniform_color([1, 0.706, 0])
       return pcd


def label2pcd(label, calibration_t, calibration_euler, scale_X, scale_Y, T_tracking,skeletonize=False,original_image=None):

    if skeletonize:
        label_skeletonized=skimage.morphology.skeletonize(label).astype(np.uint8)*255

    # extract_partial_bone_with_gradient(label,label_skeletonized,get_contour_mask(label), True)
    T_scale = np.eye(4)
    T_scale[0, 0] = scale_X
    T_scale[1, 1] = scale_Y

    rows_index, cols_index = np.where(abs(label_skeletonized) >0)
    if len(rows_index) < 2:
        return None
    # origin=np.array([[0,0,0,1]]) # for origin, it's (0,0,), for other pixels, we consider center of each pixel
    xyz_pixels = np.stack([cols_index+0.5, rows_index+0.5, np.zeros_like(rows_index), np.ones_like(rows_index)], axis=1).T

    # important_points=np.concatenate([origin,four_corners]).T
    calibrationT = vectorToMatrix(calibration_t, calibration_euler)
    T_img_2_world=T_tracking @ calibrationT @ T_scale
    xyz_world_with_features=np.ones([3+1+3,xyz_pixels.shape[1]])
    xyz_world_with_features[:4,:] = (T_img_2_world @ xyz_pixels)
    xyz_world_with_features[3,:]=intensities
    return xyz_world_with_features.T

def numpy_array_to_pcd(points,visualization=False):
    """
        Visualize a DataFrame containing x, y, z coordinates as a 3D point cloud using Open3D.

        Args:
        df (pd.DataFrame): DataFrame containing the columns 'x', 'y', 'z'.
        """
    # Check if the DataFrame contains the required columns

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Create a visualization window
    if visualization:
        o3d.visualization.draw_geometries([pcd])  # Destroy the window after closing
    return pcd