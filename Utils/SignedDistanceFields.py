import copy

import numpy as np
from scipy.ndimage import binary_fill_holes as nd_binary_fill_holes, distance_transform_edt
from skimage.morphology import closing, square,disk,erosion
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import cv2
from Utils.generalCV import *
from scipy.ndimage import convolve

def compute_intersection_points(intensities):
    i=0
    res=[]
    while i<len(intensities):
        if intensities[i]>0:
            res.append(i)
            while i<len(intensities) and intensities[i]>0 :
                i+=1
        else:
            i+=1
    return res

def generate_contour_and_mask(full_label):
    out=copy.deepcopy(full_label)
    out=connect_close_endpoints(out,10)
    if out is None:
        return None,None
    left_intersections=compute_intersection_points(out[:,0])
    right_intersections = compute_intersection_points(out[:, -1])

    bottom_intersections = compute_intersection_points(out[-1, :])
    if len(left_intersections) == 1 and len(bottom_intersections) == 1:
        out[left_intersections[0]:, 0] = 255
        out[-1, :bottom_intersections[0]] = 255

    if len(right_intersections) == 1 and len(bottom_intersections) == 1:
        out[right_intersections[0]:, -1] = 255
        out[-1, bottom_intersections[0]:] = 255
    if len(left_intersections) == 1 and len(right_intersections) == 1 and len(bottom_intersections)==0:
        out[left_intersections[0]:, 0] = 255
        out[right_intersections[0]:, -1] = 255
        out[-1, :] = 255
    if len(left_intersections)==2:
        out[left_intersections[0]:left_intersections[1],0]=255
    if len(right_intersections) == 2:
        out[right_intersections[0]:right_intersections[1], -1] = 255
    if len(bottom_intersections)==2:
        out[-1,bottom_intersections[0]:bottom_intersections[1]]=255
    if len(bottom_intersections) == 3 and len(right_intersections) == 1:
        out[-1, bottom_intersections[0]:] = 255
        out[right_intersections[0]:, -1] = 255
    if len(left_intersections)==3 and len(bottom_intersections)==1:
        out[-1,:bottom_intersections[0]]=255
        out[left_intersections[0]:,0]=255
    if len(left_intersections)==0 and len(bottom_intersections)==1 and len(right_intersections)==2:
        out[-1,bottom_intersections[0]:]=255
        out[right_intersections[0]:,-1]=255
    if len(left_intersections) == 0 and len(bottom_intersections) == 2 and len(right_intersections) == 1:
        out[-1, bottom_intersections[0]:] = 255
        out[right_intersections[0]:, -1] = 255
    if len(left_intersections)==1 and len(bottom_intersections)==2 and len(right_intersections)==0:
        out[-1, :bottom_intersections[-1]] = 255
        out[left_intersections[0]:, 0] = 255

    # print(f"left: {len(left_intersections)}, bottom: {len(bottom_intersections)}, right: {len(right_intersections)}")
    out = nd_binary_fill_holes(out)
    out=out*255
    out=out.astype(np.uint8)
    if ((out.sum()==full_label.sum())): # wrong labels
        return None,None
    # cv2.imshow("mask",out)
    refined_label=get_contour_of_binary_image(out)

    refined_label[:, 0] = full_label[:, 0]
    refined_label[:, -1] = full_label[:, -1]
    refined_label[-1, :] = full_label[-1, :]
    # refined_label = dilate_mask(refined_label, 3)

    return refined_label,out
from scipy.ndimage import gaussian_filter

def generate_partial_bone(label, mask,image):

    # Find coordinates of contour points
    partial_bone = np.zeros_like(label)
    connected_components,index_components=find_connected_components(label,size_threshold=100)
    height = label.shape[0]
    for index_component in index_components:
        partial_bone_sub_components=np.zeros_like(label)

        contour_points = np.argwhere(connected_components == index_component)


        for row,col in contour_points:
            if mask[max(row-30,0),col]>mask[min(row+30,height-1),col]:
                partial_bone_sub_components[row,col]=255

        # partial_bone_sub_components=close_image_gaps(partial_bone_sub_components,5)
        partial_bone_sub_components=keep_top_k_largest_components(partial_bone_sub_components,1,original_image=image,intensity_sum=True,area=False)
        partial_bone[partial_bone_sub_components>0]=255
    i = 0
    while i < height:
        if partial_bone[i, 0] > 0:
            end_idx = i + 1
            while end_idx < height and partial_bone[end_idx, 0] > 0:
                end_idx += 1
            if end_idx - i > 5:
                partial_bone[i + 5:end_idx, 0] = 0
            # print(f"end_idx:{end_idx - i}")
            i += end_idx
        else:
            i += 1
    i = 0
    while i < height:
        if partial_bone[i, -1] > 0:
            end_idx = i + 1
            while end_idx < height and partial_bone[end_idx, -1] > 0:
                end_idx += 1
            if end_idx - i > 3:
                partial_bone[i + 5:end_idx, -1] = 0
            # print(f"end_idx:{end_idx - i}")
            i += end_idx
        else:
            i += 1
         # else:
         #   i+=1



    return partial_bone



def compute_distance_2Dmap(full_label,truncated=200):
    """
    Compute the distance map for a label image where each pixel's value
    is the distance to the closest white (value 255) pixel.

    Args:
    label_image (numpy.ndarray): A label image where white pixels (255) represent areas of interest.

    Returns:
    numpy.ndarray: Distance map where each pixel's value is the distance to the nearest white pixel (255).
    """
    # Ensure the image is boolean where True (1) is 255 and False (0) is the background.
    refined_contour,mask=generate_contour_and_mask(full_label)
    if refined_contour is None:
        return None,None

    # Compute the distance transform: distance from non-object (0) pixels to the nearest object (1) pixel
    distance_map = distance_transform_edt(~refined_contour)
    distance_map[distance_map > truncated] = truncated
    # Create the signed distance field where inside mask distances are negative
    distance_map[mask > 0] *= -1

    return refined_contour,distance_map


def visualize_distance_map(image, label, distance_map):
    """
    Visualize a distance map with a custom color scheme in a given subplot.
    Negative values are displayed in red, positive values in blue. The transition around zero blends between red and blue.

    Args:
    distance_map (numpy.ndarray): A numpy array containing the distance values.
    ax (matplotlib.axes.Axes, optional): The axes on which to draw the image. If None, uses the current axes.
    """
    # Create a custom colormap from red to blue, passing through a neutral color at zero
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Label image
    axes[1].imshow(label, cmap='gray')
    axes[1].set_title('Label Image')
    axes[1].axis('off')

    neutral_color=0.8
    colors = [(1, 0, 0), (neutral_color, neutral_color, neutral_color), (0, 0, 1)]  # Red, Light Gray, Blue
    cmap_name = 'my_colormap'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)



    # Ensure zero is exactly at the middle of the color range by setting the normalization
    max_abs_value = np.max(np.abs(distance_map))
    norm = Normalize(vmin=-max_abs_value, vmax=max_abs_value)

    # Determine the axes for plotting
    if axes[2] is None:
        axes[2] = plt.gca()

    # Plotting on the specified axes
    im = axes[2].imshow(distance_map, cmap=cmap, interpolation='nearest', norm=norm)
    cbar = plt.colorbar(im, ax=axes[2])
    cbar.set_label('Distance Values')
    axes[2].set_title('Distance Map Visualization')

    opencv_image = plt_to_opencv(fig)
    plt.close(fig)
    return opencv_image

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
def plt_to_opencv(fig):
    """
    Convert a Matplotlib figure to an image format that can be displayed with OpenCV.
    """
    # Convert the Matplotlib figure to an RGBA buffer.
    canvas = FigureCanvas(fig)
    canvas.draw()
    rgba_image = np.array(canvas.renderer.buffer_rgba())
    # Convert RGBA to BGR
    bgr_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
    return bgr_image



def extract_partial_bone_with_gradient(SDF, label,debug_F=False):

    grad_x, grad_y = compute_gradient(SDF,59)

    # Step 3: Normalize the gradient to get unit vectors (normals)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    unit_grad_x = grad_x / (magnitude + 1e-8)  # add epsilon to avoid division by zero
    unit_grad_y = grad_y / (magnitude + 1e-8)

    # Step 4: Mask the normals where label is 255

    # Get the coordinates where label is 255
    y, x = np.where(label==255)

    # Apply step size to reduce the density of the arrows
    if debug_F:
        step=20
        x = x[::step]
        y = y[::step]
    dx = unit_grad_x[y, x]



    dy = unit_grad_y[y, x]

    # Step 5: Create an image for visualization
    if debug_F:
        output_img=np.zeros_like(label)
        output_img = cv2.cvtColor(output_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    partial_bone_gradient_based=np.zeros_like(label)

    # Draw arrows
    angles = compute_angles(grad_x, grad_y, (0, -1))
    angle_threshold = 50
    if debug_F:
        for (x_i, y_i, dx_i, dy_i) in zip(x, y, dx, dy):
            if angles[y_i,x_i]<=angle_threshold:
                partial_bone_gradient_based[y_i,x_i]=255
                if debug_F:
                    pt1 = (int(x_i), int(y_i))  # Start point (y, x)
                    pt2 = (1*int(x_i + dx_i * 50), 1*int(y_i + dy_i * 50))  # End point (y + v, x + u)
                    cv2.arrowedLine(output_img, pt1, pt2, (0, 0, 255), tipLength=0.5,thickness=1)  # Red arrows

    angles[angles<=angle_threshold]=0
    angles[angles>angle_threshold]=255
    angles=255-angles


    # Display the result using OpenCV


    if debug_F:
        cv2.imshow("partial bone", angles * label)
        cv2.imshow('Normal Vectors', output_img)
    return angles * (label>0)

def compute_gradient(SDF, kernel_size):
    """
    Compute the gradient using a specified kernel size.

    Parameters:
    - SDF: 2D numpy array, the signed distance function.
    - kernel_size: Integer, size of the kernel to use for gradient computation.

    Returns:
    - grad_x: Gradient in the x-direction.
    - grad_y: Gradient in the y-direction.
    """
    # Create kernels for gradient computation
    SDF[SDF<0]=-1
    SDF[SDF>0]=1
    kernel_y = np.zeros((kernel_size, kernel_size))
    kernel_x = np.zeros((kernel_size, kernel_size))

    mid = kernel_size // 2
    kernel_y[:, mid] = np.arange(-mid, mid + 1)
    kernel_x[mid, :] = np.arange(-mid, mid + 1)

    # Normalize kernels
    kernel_x /= (mid * (mid + 1))
    kernel_y /= (mid * (mid + 1))
    kernel_x*=-1
    kernel_y*=-1

    # Convolve SDF with the kernels
    grad_x = convolve(SDF, kernel_x)
    grad_y = convolve(SDF, kernel_y)

    return grad_x, grad_y

def angle_between_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Compute the dot product
    dot_product = np.dot(v1, v2)

    # Compute the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Compute the cosine of the angle
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)

    # Compute the angle in radians
    angle_radians = np.arccos(cos_theta)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def compute_angles(grad_x, grad_y, target_vector=(0, 1)):
    """
    Compute the angle between the (grad_x, grad_y) vector and the target vector (0, 1)
    for each pixel in the grad_x and grad_y arrays.

    Parameters:
    - grad_x: 2D numpy array representing the x component of the gradient.
    - grad_y: 2D numpy array representing the y component of the gradient.
    - target_vector: Tuple representing the target vector (default is (0, 1)).

    Returns:
    - angles: 2D numpy array of the same shape as grad_x containing the angles in degrees.
    """
    # Convert the target vector to a numpy array
    target_vector = np.array(target_vector)

    # Compute the dot product between (grad_x, grad_y) and the target vector (0, 1)
    dot_product = grad_y * target_vector[1]  # Only y-component matters as target vector is (0, 1)

    # Compute the magnitude of the gradient vectors
    magnitude_grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Compute the magnitude of the target vector (which is 1 for (0, 1))
    magnitude_target = np.linalg.norm(target_vector)

    # Compute the cosine of the angle
    cos_theta = dot_product / (magnitude_grad * magnitude_target + 1e-8)  # Add epsilon to avoid division by zero

    # Ensure cos_theta is within the valid range for arccos due to possible numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Compute the angle in radians
    angle_radians = np.arccos(cos_theta)

    # Convert the angle to degrees
    angles = np.degrees(angle_radians)

    return angles
