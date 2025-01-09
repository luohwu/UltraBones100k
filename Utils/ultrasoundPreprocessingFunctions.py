import shutil
import time

import pandas as pd
import os

from Utils.fit2pcd import compute_inverse_transformation
import  numpy as np
from scipy.spatial.transform import  Rotation as R
import phasepack as pp

def compute_phase_symmetry(image,num_scales=2,min_wavelength=25,sigma_on_f=15):
    # Load your ultrasound image (grayscale)

    # Parameters for phase symmetry
    # num_scales = 6  # Number of scales (orientations)
    # min_wavelength = 25  # Minimum wavelength of the sinusoidal functions
    # sigma_on_f = 15  # Bandwidth parameter

    # Calculate phase symmetry
    results = pp.phasesym(image, nscale=num_scales, minWaveLength=min_wavelength, sigmaOnf=(1/sigma_on_f))
    psym = results[0]  # Extracting only the phase symmetry part

    # cv2.imshow('a', psym)
    # cv2.waitKey(0)
    return psym

def preprocessData(timeThreshold,data_folder,temporal_offset=0,tracking_error_tol=0.3):

    # if os.path.isfile(os.path.join(data_folder, "trackingProcessed.csv")):
    #     dfTracking = pd.read_csv(os.path.join(data_folder, "trackingProcessed.csv"))
    # else:
    #     dfTracking = preprocess_tracking_data(os.path.join(data_folder, "tracking.csv"))
    #
    # print("finish pre-processing tracking data")
    #
    # if os.path.isfile(os.path.join(data_folder, "sweepProcssed.csv")):
    #     dfUS = pd.read_csv(os.path.join(data_folder, "sweepProcssed.csv"))
    # else:
    #     dfUS = preprocessSweepData("sweep.csv", dfTracking,timeThreshold)
    print(f"pre-processing: {data_folder}")
    dfTracking = preprocess_tracking_data(os.path.join(data_folder, "tracking.csv"),data_folder,tracking_error_tol=tracking_error_tol)
    # print("finish pre-processing tracking data")
    dfUS = preprocessSweepData("sweep.csv", dfTracking, timeThreshold,data_folder,temporal_offset)
    # print(f"finish pre-processing US data")

    return dfTracking,dfUS


def preprocess_tracking_data(filename,data_folder,tracking_error_tol=0.3):
    df=pd.read_csv(filename)
    df_processed=pd.DataFrame(columns=df.columns)
    for index in range(len(df) - 1):
        if df.loc[index,'timestamp']==df.loc[index+1,'timestamp']:
            if df.iloc[index]['anchor']==True and df.iloc[index+1]['anchor']==False:

                row_anchor=df.iloc[index]
                row_probe=df.iloc[index+1]
            elif df.iloc[index+1]['anchor']==True and df.iloc[index]['anchor']==False:
                row_anchor = df.iloc[index+1]
                row_probe = df.iloc[index ]
            else:
                continue
            if 'error' in df_processed.columns:
                if row_probe['error']>tracking_error_tol or row_anchor['error']>tracking_error_tol:
                    continue

            T_probe = np.array([
                [row_probe['r11'], row_probe['r12'], row_probe['r13'], row_probe['x']],
                [row_probe['r21'], row_probe['r22'], row_probe['r23'], row_probe['y']],
                [row_probe['r31'], row_probe['r32'], row_probe['r33'], row_probe['z']],
                [0, 0, 0, 1]
            ])
            T_anchor = np.array([
                [row_anchor['r11'], row_anchor['r12'], row_anchor['r13'], row_anchor['x']],
                [row_anchor['r21'], row_anchor['r22'], row_anchor['r23'], row_anchor['y']],
                [row_anchor['r31'], row_anchor['r32'], row_anchor['r33'], row_anchor['z']],
                [0, 0, 0, 1]
            ])
            T_target_to_anchor = compute_inverse_transformation(T_anchor) @ T_probe
            if 'error' in df_processed.columns:
                df_processed.loc[len(df_processed)] = [row_probe['timestamp'], row_probe['geometry_id'],
                                                       row_probe['anchor'], T_target_to_anchor[0][3],
                                                       T_target_to_anchor[1][3],
                                                       T_target_to_anchor[2][3]
                    , T_target_to_anchor[0][0], T_target_to_anchor[0][1], T_target_to_anchor[0][2]
                    , T_target_to_anchor[1][0], T_target_to_anchor[1][1], T_target_to_anchor[1][2]
                    , T_target_to_anchor[2][0], T_target_to_anchor[2][1], T_target_to_anchor[2][2],row_probe['error']+row_anchor['error']]
            else:
                df_processed.loc[len(df_processed)] = [row_probe['timestamp'], row_probe['geometry_id'],
                                                       row_probe['anchor'], T_target_to_anchor[0][3],
                                                       T_target_to_anchor[1][3],
                                                       T_target_to_anchor[2][3]
                    , T_target_to_anchor[0][0], T_target_to_anchor[0][1], T_target_to_anchor[0][2]
                    , T_target_to_anchor[1][0], T_target_to_anchor[1][1], T_target_to_anchor[1][2]
                    , T_target_to_anchor[2][0], T_target_to_anchor[2][1], T_target_to_anchor[2][2]]
    df_processed.to_csv(os.path.join(data_folder, "trackingProcessed.csv"),index=False)

    # timestamp_diff=df_processed['timestamp'].diff()
    # print(f"mean of timestamp diff: {timestamp_diff[1:].mean()}")
    # print(df_process.head())
    return df_processed
def get_interpolated_value(target_time,df:pd.DataFrame,timeThreshold):

    right_index=df[df['timestamp']>target_time].index[0]
    if right_index<0:
        return None,None

    right_time=df.iloc[right_index]['timestamp']

    index_x = df.columns.get_loc('x')
    index_r11 = df.columns.get_loc('r11')

    right_t=df.iloc[right_index][index_x:index_x+3].to_numpy()
    right_rotation_matrix=df.iloc[right_index][index_r11:index_r11+9].to_numpy().reshape(3,3)
    right_rotation_vector=R.from_matrix(right_rotation_matrix).as_euler('xyz',degrees=True)


    right_vector=np.concatenate((right_t,right_rotation_vector))



    left_index=right_index-1
    left_time=df.iloc[left_index]['timestamp']
    left_t=df.iloc[left_index][index_x:index_x+3].to_numpy()
    left_rotation_matrix=df.iloc[left_index][index_r11:index_r11+9].to_numpy().reshape(3,3)
    left_rotation_vector=R.from_matrix(left_rotation_matrix).as_euler('xyz',degrees=True)
    left_vector = np.concatenate((left_t , left_rotation_vector))

    if target_time-left_time>timeThreshold or right_time-target_time>timeThreshold:
        return None,None


    if left_time==target_time:
        interpolated_vec= left_vector
    else:
        total_distance=right_time-left_time
        interpolated_vec= ((target_time-left_time)/total_distance*right_vector+(right_time-target_time)/total_distance*left_vector)

    left_tracking_error=df.iloc[left_index]['error'] if 'error' in df.columns else None

    return interpolated_vec.tolist(),left_tracking_error

def preprocessSweepData(filename,dfTracking,timeThreshold,data_folder,temporal_offset=0):


    time.sleep(1.5)
    minTimeTracking=dfTracking.loc[0,'timestamp']
    maxTimeTracking=dfTracking.loc[len(dfTracking)-1,'timestamp']
    pathSweep=os.path.join(data_folder,filename)
    dfSweep = pd.read_csv(pathSweep)
    dfSweep=dfSweep[dfSweep['timestamp']>minTimeTracking]
    dfSweep=dfSweep[dfSweep['timestamp']<maxTimeTracking]
    dfSweepProcessed=pd.DataFrame(columns=dfSweep.columns)
    dfSweepProcessed['x'] = np.nan
    dfSweepProcessed['y'] = np.nan
    dfSweepProcessed['z'] = np.nan
    dfSweepProcessed['euler_x'] = np.nan
    dfSweepProcessed['euler_y'] = np.nan
    dfSweepProcessed['euler_z'] = np.nan
    if 'error' in dfTracking:
        dfSweepProcessed['error'] = np.nan
    for index in range(len(dfSweep)):
        row=dfSweep.iloc[index]
        interpolatedTracking,tracking_error=get_interpolated_value(row['timestamp']+temporal_offset,dfTracking,timeThreshold)
        if interpolatedTracking is not None:
            if 'error' in dfTracking:
                dfSweepProcessed.loc[len(dfSweepProcessed)]=row.tolist()+interpolatedTracking+[tracking_error]
            else:
                dfSweepProcessed.loc[len(dfSweepProcessed)] = row.tolist() + interpolatedTracking
    dfSweepProcessed.to_csv(os.path.join(data_folder, "sweepProcessed.csv"),index=False)
    # print(dfSweepProcessed.head())
    return dfSweepProcessed