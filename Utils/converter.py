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

def sweepFilesToPCD(files:list,temporalOff,calibrationT,spacing=None,target_depth=[0,1],labeled=False):
       pcd=o3d.geometry.PointCloud()
       for file in files:
              sweep=imfusion.io.open(file)[0]
              sweep.tracking().temporalOffset=temporalOff
              sweep.tracking().calibration=calibrationT
              sweep_np=sweep.numpy()
              for i in range(sweep_np.shape[0]):
                     img_imf=sweep.img(i)
                     if spacing:
                            img_imf.spacing = spacing
                     img_np=np.array((sweep_np[i]))
                     non_zero_pixels=img_np[img_np>0]
                     # cv2.imshow("original image",img_np)
                     # cv2.waitKey(0)
                     # _,img_thresholded=cv2.threshold(img_np,np.percentile(non_zero_pixels,95),255,cv2.THRESH_BINARY)
                     # cv2.imshow("image",img_thresholded)
                     # cv2.waitKey(0)
                     if labeled==False:
                            pass
                            # rows_index, cols_index = np.where(img_thresholded > 0)
                     else:
                            rows_index, cols_index,_ = np.where(img_np > 0)
                     index_within_target_depth=(rows_index<target_depth[1]*img_imf.height) * ((rows_index>target_depth[0]*img_imf.height))
                     rows_index=rows_index[index_within_target_depth]
                     cols_index=cols_index[index_within_target_depth]
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


if __name__=='__main__':
       sweep_files=['./ForwardSweep.imf','./SideSweep.imf']
       sweep_files = [
              # '../PointCloudConverter/mockup.imf',
              '../PointCloudConverter/train1.imf',
              '../PointCloudConverter/train2.imf',
       ]
       # sweep_files=['D:\ImFusion projects\sweeps\mockup1.imf',
       #              'D:\ImFusion projects\sweeps\mockup2.imf',
       #              'D:\ImFusion projects\sweeps\mockup3.imf',
       #              ]
       temporalOff=0.0646
       calibration_t=[0,0,48.5]
       calibration_euler=[90, 0, 180]
       calibrationT=vectorToMatrix(calibration_t,calibration_euler)
       pcd=sweepToPCD(sweep_files,temporalOff,calibrationT,target_depth=[0.2,0.5])
       o3d.visualization.draw_geometries([pcd])
       o3d.io.write_point_cloud('../Ultrasound/Ultrasound_SCMI_L18/PointCloudConverter/pcd.ply', pcd)


