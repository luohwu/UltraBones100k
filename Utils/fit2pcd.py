import numpy as np
from scipy.spatial.transform import  Rotation as R

def vectorToMatrix(t, euler_xyz):
    T = np.eye(4)
    r = R.from_euler('xyz', euler_xyz, degrees=True)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = t
    return T

##Based on Arun et al., 1987
def fit_2_point_cloud(p1_moving, p2_fixed):
    #Writing points with rows as the coordinates
    # p1_moving = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    # p2_fixed = np.array([[0, 0, 1], [1, 0, 1], [0, 0, 2]]) #Approx transformation is 90 degree rot over x-axis and +1 in Z axis

    #Take transpose as columns should be the points
    p1 = np.array(p1_moving).transpose()
    p2 = np.array(p2_fixed).transpose()
    #
    # p1 = p1.transpose()
    # p2 = p2.transpose()

    #Calculate centroids
    p1_c = np.mean(p1, axis = 1).reshape((-1,1)) #If you don't put reshape then the outcome is 1D with no rows/colums and is interpeted as rowvector in next minus operation, while it should be a column vector
    p2_c = np.mean(p2, axis = 1).reshape((-1,1))


    #Subtract centroids
    q1 = p1-p1_c
    q2 = p2-p2_c

    #Calculate covariance matrix
    H=np.matmul(q1,q2.transpose())

    #Calculate singular value decomposition (SVD)
    U, X, V_t = np.linalg.svd(H) #the SVD of linalg gives you Vt

    #Calculate rotation matrix
    R = np.matmul(V_t.transpose(),U.transpose())

    assert np.allclose(np.linalg.det(R), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

    #Calculate translation matrix
    t = p2_c - np.matmul(R,p1_c)

    #Check result
    result = t + np.matmul(R,p1)
    if np.allclose(result,p2):
        print("transformation is correct!")
    else:
        print("transformation is wrong...")
    T=np.eye(4)
    T[:3,:3]=R
    T[:3,3]=t.T
    return T

def compute_inverse_transformation(transformation_matrix):
    """
    Compute the inverse of a 4x4 transformation matrix.

    Args:
    transformation_matrix (np.array): A 4x4 transformation matrix.

    Returns:
    np.array: Inverse of the transformation matrix.
    """
    if transformation_matrix.shape != (4, 4):
        raise ValueError("Input must be a 4x4 matrix.")

    # Extracting R and t from the transformation matrix
    R = transformation_matrix[:3, :3]  # Top-left 3x3 submatrix for rotation
    t = transformation_matrix[:3, 3]  # Top-right 3x1 subvector for translation

    # Computing the inverse transformation matrix
    R_transpose = R.T
    t_transpose = -np.dot(R_transpose, t)
    T_inv = np.vstack([np.column_stack([R_transpose, t_transpose]), [0, 0, 0, 1]])

    return T_inv

if __name__ == "__main__":
    T=np.eye(4)
    print(compute_inverse_transformation(T))
