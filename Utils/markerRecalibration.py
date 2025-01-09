import numpy as np
def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def read_geometry_file(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    result=np.zeros([4,3])
    result[0][0]=float(lines[1][2:])
    result[0][1]=float(lines[2][2:])
    result[0][2]=float(lines[3][2:])
    result[1][0]=float(lines[5][2:])
    result[1][1]=float(lines[6][2:])
    result[1][2]=float(lines[7][2:])
    result[2][0]=float(lines[9][2:])
    result[2][1]=float(lines[10][2:])
    result[2][2]=float(lines[11][2:])
    result[3][0]=float(lines[13][2:])
    result[3][1]=float(lines[14][2:])
    result[3][2]=float(lines[15][2:])
    return result.transpose()


if __name__=='__main__':

    id=1380
    original_file_path="D:/spryTrack SDK x64/data/geometry1480.ini"
    original_geometry_order=[0,1,2,3]
    recalibrated_file_path="C:/Users/luoho/AppData/Local/Atracsys/PassiveTrackingSDK/geometry148000.ini"
    recalibrated_geometry_order=[0,1,2,3]
    read_geometry_file(original_file_path)

    # recalibrated points by Atracsys
    # mind the order of points
    pointCloudA=read_geometry_file(recalibrated_file_path)[:,recalibrated_geometry_order]

    # original points from blender
    pointCloudB=read_geometry_file(original_file_path)[:,original_geometry_order]

    R,t=rigid_transform_3D(pointCloudA,pointCloudB)
    print((R@pointCloudA+t).transpose())
    print("-"*100)
    result=(R@pointCloudA+t).transpose()

    print("[fiducial0]")
    print(f"x={result[0][0]}")
    print(f"y={result[0][1]}")
    print(f"z={result[0][2]}")
    print("[fiducial1]")
    print(f"x={result[1][0]}")
    print(f"y={result[1][1]}")
    print(f"z={result[1][2]}")
    print("[fiducial2]")
    print(f"x={result[2][0]}")
    print(f"y={result[2][1]}")
    print(f"z={result[2][2]}")
    print("[fiducial3]")
    print(f"x={result[3][0]}")
    print(f"y={result[3][1]}")
    print(f"z={result[3][2]}")
    print("[geometry]")
    print("count=4")
    print(f"id={id}")

