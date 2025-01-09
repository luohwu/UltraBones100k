import configparser

import numpy as np
from io import StringIO

def parse_geometry_file_to_numpy(file_path: str) -> np.ndarray:
    """
    Parses the geometry file of an optical marker to extract XYZ coordinates of all fiducials
    and returns a numpy array where each column is a point's xyz.

    Parameters:
    - file_path: str, the path to the geometry file.

    Returns:
    - A numpy array where each column represents the xyz coordinates of a fiducial.
    """
    # Initialize configparser
    config = configparser.ConfigParser()
    config.read(file_path)

    # List to store coordinates of all fiducials
    points = []

    # Iterate through each section in the config if it starts with 'fiducial'
    for section in config.sections():
        if section.startswith('fiducial'):
            point = [
                float(config[section]['x']),
                float(config[section]['y']),
                float(config[section]['z']),
                1
            ]
            points.append(point)

    # Convert list of points to a numpy array and transpose it
    # so that each column represents a point's xyz coordinates
    points_array = np.array(points).T

    return points_array



def simulate_geometry_content_creation(points_array, geometry_id):
    config = configparser.ConfigParser()

    for i, (x, y, z) in enumerate(points_array.T):
        section_name = f'fiducial{i}'
        config[section_name] = {'x': str(x), 'y': str(y), 'z': str(z)}

    config['geometry'] = {'count': str(points_array.shape[1]), 'id': str(geometry_id)}

    # Writing to a string to simulate file content
    with StringIO() as output:
        config.write(output)
        content = output.getvalue()

    return content

if __name__=='__main__':

    # modify T
    # modify recalibrated xyz file path
    # modify target geometry id

    T=np.array([
        [-0.8463,0.5326,0.0125,25.14],
        [0.02612,0.06492,-0.9975,21.9],
        [-0.5321,-0.8439,-0.06885,-6.817],
        [0,0,0,1]
    ])
    np.set_printoptions(5,suppress=True)
    # Use the function to parse the uploaded file and get the coordinates as a numpy array


    file_path = "geometryFiles/geometry301800.ini"
    xyz_recalibrated = parse_geometry_file_to_numpy(file_path)
    print(xyz_recalibrated)
    print(T@xyz_recalibrated)
    xyz_registered=T@xyz_recalibrated

    print("="*80)
    print(simulate_geometry_content_creation(xyz_registered[:3,:],30180))
