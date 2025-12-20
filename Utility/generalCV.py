import copy

import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image

from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
def overlap_image_with_label( image, mask1,weight_mask=0.8):
    # Create a colored version of mask1 where each value is mapped to a color
    colored_mask1 = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)

    # Define colors for each mask value (0, 50, 100, 150, 200, 250)
    colors = {
        0: [0, 0, 0],  # Black for 0 (usually for background)
        1: [0, 255, 0],  # Green for 50
        4: [0, 0, 255],  # Blue for 100
        5: [255, 255, 0],  # Cyan for 200
        9: [255, 0, 0],  # Red for 150
        255: [0, 255, 0]  # Magenta for 250
    }

    # Apply colors to each area based on mask1's value
    for value, color in colors.items():
        colored_mask1[mask1 == value] = color

    # Convert the single-channel image to 3-channel for overlaying
    if len(image.shape) == 2:  # Check if the image is grayscale
        image_3_channel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_3_channel = image.copy()

    # Overlay the colored masks on the image
    # Adjust the weight to control the transparency of the overlays
    overlay_image = cv2.addWeighted(image_3_channel, 1, colored_mask1, weight_mask, 0)

    return overlay_image


def overlap_image_with_label_two_classes( image, mask1,weight_mask=0.9,threshold=50):
    # Create a colored version of mask1 where each value is mapped to a color
    colored_mask1 = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
    # mask1 = gaussian_filter(image, sigma=0.5) * mask1
    image_smoothed=gaussian_filter(copy.deepcopy(image),sigma=4)
    # image_smoothed=image
    mask1[np.logical_and(mask1>0,image_smoothed<threshold)]=1
    mask1[np.logical_and(mask1>0,image_smoothed>=threshold)]=4
    # mask1[mask1>=threshold]=4
    # Define colors for each mask value (0, 50, 100, 150, 200, 250)
    colors = {
        0: [0, 0, 0],  # Black for 0 (usually for background)
        4: [0, 255, 0],
        1: [255, 0, 0],
        5: [255, 255, 0],  # Cyan for 200
        9: [255, 0, 0],  # Red for 150
        255: [0, 255, 0]  # Magenta for 250
    }

    # Apply colors to each area based on mask1's value
    for value, color in colors.items():
        colored_mask1[mask1 == value] = color

    # Convert the single-channel image to 3-channel for overlaying
    if len(image.shape) == 2:  # Check if the image is grayscale
        image_3_channel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_3_channel = image.copy()

    # Overlay the colored masks on the image
    # Adjust the weight to control the transparency of the overlays
    overlay_image = cv2.addWeighted(image_3_channel, 1, colored_mask1, weight_mask, 0)

    return overlay_image



def keep_top_k_largest_components(binary_image, k,area=False,original_image=None,area_threshold=None,intensity_sum=False):
    """
    Keep only the top k largest connected components in a binary image based on their dimensions (height * width).

    Parameters:
    - binary_image: A 2D numpy array with binary values (0 and 255)

    Returns:
    - clean_image: The image with only the k largest components retained based on dimensions
    """
    # Ensure the binary image is of type uint8
    binary_image = binary_image.astype(np.uint8)

    # Find all connected components (also called blobs or labels)
    num_labels, labels_im = cv2.connectedComponents(binary_image)

    # Dictionary to hold each component and its dimensions
    dimensions = {}

    # Loop through all found components
    for label in range(1, num_labels):  # Start from 1 to ignore the background label
        # Create a mask where the current label is located
        component_mask = (labels_im == label)

        # Find the bounds of the component
        if area==False:

            dimensions[label] = original_image[component_mask].sum()
        else:
            ys, xs = np.where(component_mask)
            height = ys.max() - ys.min() + 1
            width = xs.max() - xs.min() + 1

            # Calculate the dimension as height * width
            dimension = height * width

            # Store the dimensions with corresponding label
            dimensions[label] = dimension

    # Find labels of the top k largest components based on their dimensions
    # Sort dimensions items by the dimension, reverse to get largest first, and pick top k
    largest_components = sorted(dimensions, key=dimensions.get, reverse=True)[:k]
    if not area_threshold is None:
        largest_components=[item for item in largest_components if dimensions[item]>area_threshold]
    # print([dimensions[item] for item in largest_components])

    # Create an output image that will hold the largest components
    clean_image = np.zeros_like(binary_image)

    # Loop through labels of the largest components and copy them to the output image
    #

    if original_image is None or intensity_sum==True:
        for idx, label in enumerate(largest_components):
            component_mask = (labels_im == label)
            clean_image[component_mask] = 255
    else:
        label_infos=[]
        for idx,label in enumerate(largest_components):
            label_info={}
            component_mask = (labels_im == label)
            intensities = original_image[component_mask]
            intensities_mean = intensities.mean()
            intensities_std = intensities.std()
            intensities_95p = np.percentile(intensities, 95)

            label_info['component_mask']=component_mask
            label_info['intensities_mean'] = intensities_mean
            label_info['intensities_std'] = intensities_std
            label_info['intensities_95p'] = intensities_95p
            label_info['intensities_score_total'] =intensities_mean+intensities_std+intensities_95p
            label_infos.append(label_info)
        label_infos = sorted(label_infos, key=lambda x: x['intensities_score_total'], reverse=True)
        for idx,label_info in enumerate(label_infos):
            if idx==0:
                clean_image[label_info['component_mask']]=255
            else:
                if label_info['intensities_mean'] >130 and label_info['intensities_std']>50 and label_info['intensities_95p'] >200:
                    clean_image[label_info['component_mask']] = 255



    return clean_image

def find_connected_components(binary_image, size_threshold=1000):
    """
    Identify connected components in a binary image and ignore components with size less than a specified threshold.

    Parameters:
    - binary_image (numpy.array): A 2D numpy array with binary values (0 and 255)
    - size_threshold (int): The minimum size threshold for connected components to be considered

    Returns:
    - numpy.array: A label map where each connected component with size above the threshold is marked with a unique index,
                   and smaller components are ignored (set to 0).
    """
    # Ensure the input is a binary image of type uint8
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8)

    # Apply the connectedComponents function
    num_labels, labels_im = cv2.connectedComponents(binary_image)

    # Array to hold the size of each component
    component_sizes = np.zeros(num_labels, dtype=int)

    # Calculate the size of each component
    for label in range(1, num_labels):  # start from 1 to ignore the background
        component_sizes[label] = np.sum(labels_im == label)

    # Create a mask to remove components that don't meet the size threshold
    for label, size in enumerate(component_sizes):
        if size < size_threshold:
            labels_im[labels_im == label] = 0

    # Optional: Renumber labels to be continuous
    unique_labels = np.unique(labels_im)
    # new_labels = np.arange(unique_labels.size)
    # remap_dict = dict(zip(unique_labels, new_labels))
    # labels_remapped = np.vectorize(remap_dict.get)(labels_im)
    # labels_remapped=labels_remapped.astype(np.uint8)

    return labels_im,unique_labels[unique_labels!=0] #0 is background

# def find_connected_components(binary_image):
#     """
#     Identify all connected components in a binary image.
#
#     Parameters:
#     - binary_image (numpy.array): A 2D numpy array with binary values (0 and 255)
#
#     Returns:
#     - numpy.array: A label map where each connected component is marked with a unique index,
#                    starting from 1.
#     - int: The number of unique connected components, excluding the background.
#     """
#     # Ensure the input is a binary image of type uint8
#     if binary_image.dtype != np.uint8:
#         binary_image = binary_image.astype(np.uint8)
#
#     # Apply the connectedComponents function
#     num_labels, labels_im = cv2.connectedComponents(binary_image)
#
#     # We don't need to calculate component sizes or filter them based on size.
#     # So, directly create a label map with unique indices.
#
#     # Optional: Renumber labels to be continuous
#     # This part is optional and depends on whether you need a 0-based index or not.
#     # If not required, simply return labels_im as is after the following line.
#     # Adjust the output for zero-based indexing by subtracting 1, if necessary.
#     labels_im = labels_im.astype(np.uint8)  # Ensure labels are returned as uint8.
#
#     return labels_im, num_labels - 1  # Exclude the background label from the count


def merge_images_horizontally(images):
    """
    Merge multiple images horizontally into a single image, ensuring all are converted to three channels.

    Parameters:
    - images (list of numpy.array): The list of images to merge.

    Returns:
    - numpy.array: The horizontally merged image.
    """
    # Ensure all images have three channels
    converted_images = []
    for image in images:
        if len(image.shape) == 2:  # Check if image is grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:  # Check if image is single-channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        converted_images.append(image)

    # Resize images to match the height of the first image
    target_height = converted_images[0].shape[0]
    resized_images = []
    for image in converted_images:
        # cv2.imshow("a",image)
        # cv2.waitKey(0)
        if image.shape[0] != target_height:
            scale_factor = target_height / image.shape[0]
            new_width = int(image.shape[1] * scale_factor)
            image = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
        resized_images.append(image)

    # Merge all images horizontally
    merged_image = np.hstack(resized_images).astype(np.uint8)

    return merged_image


def merge_images_vertically(images):
    """
    Merge multiple images horizontally into a single image, ensuring all are converted to three channels.

    Parameters:
    - images (list of numpy.array): The list of images to merge.

    Returns:
    - numpy.array: The horizontally merged image.
    """
    # Ensure all images have three channels
    converted_images = []
    for image in images:
        if len(image.shape) == 2:  # Check if image is grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:  # Check if image is single-channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        converted_images.append(image)

    # Resize images to match the height of the first image
    target_height = converted_images[0].shape[0]
    resized_images = []
    for image in converted_images:
        if image.shape[0] != target_height:
            scale_factor = target_height / image.shape[0]
            new_width = int(image.shape[1] * scale_factor)
            image = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
        resized_images.append(image)

    # Merge all images horizontally
    merged_image = np.vstack(resized_images)

    return merged_image



def close_image_gaps(binary_image, kernel_size=3):
    """
    Close small gaps in a binary image using morphological closing.

    Parameters:
    - binary_image: A binary image (numpy array) where the objects are 255 and the background is 0.
    - kernel_size: Size of the structuring element used for closing.

    Returns:
    - closed_image: The binary image after applying the closing operation.
    """
    # Create the structuring element
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply morphological closing
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    return closed_image

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
def enhance_bone(image,sigma=1,percentile=90):
    # Load image in grayscale and convert to float

    # Calculate Hessian matrix and its eigenvalues
    hessian = hessian_matrix(image, sigma=sigma, order='rc',use_gaussian_derivatives=False)
    eigenvalues = hessian_matrix_eigvals(hessian)

    # Eigenvalues are sorted, so ev1 is the smaller and ev2 is the larger
    ev1, ev2 = eigenvalues[0], eigenvalues[1]

    bone_mask = np.abs(ev1 - ev2) > np.percentile(np.abs(ev1 - ev2), percentile)  # Significant difference
    bone_mask=bone_mask*255

    return bone_mask.astype(np.uint8)

def unnormalize_tensor(tensor,mean=0.17475835978984833,std=0.16475939750671387):
    return tensor*std+mean

def tensor_2_opencv(img_tensor,mean=0.17475835978984833,std=0.16475939750671387):


    if mean>0:
        img_tensor=img_tensor*std+mean

    img_pil = to_pil_image(img_tensor)
    img_np = np.array(img_pil)
    if img_np.ndim == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np

def resize_image_by_scale(image, scale_width,scale_height):
    """
    Resize the image based on a scale factor. Positive scale factors enlarge the image,
    negative scale factors reduce the image size.

    Args:
    image (numpy.array): The input image array.
    scale (float): The scale factor for resizing. Must not be zero.

    Returns:
    numpy.array: The resized image array.
    """
    if scale_width == 0 or scale_height==0 or scale_height*scale_width<0:
        raise ValueError("Scale factor must not be zero and have opposite signs.")

    # Calculate the absolute scale factor
    abs_scale_width = abs(scale_width)
    abs_scale_height=abs(scale_height)

    # Determine new dimensions


    # Choose interpolation method based on the scale factor
    if scale_width > 0:
        new_width = int(image.shape[1] * abs_scale_width)
        new_height = int(image.shape[0] * abs_scale_height)
        # Use INTER_LINEAR for enlargement
        interpolation = cv2.INTER_LINEAR
    else:
        # Use INTER_AREA for reduction
        interpolation = cv2.INTER_AREA
        new_width = int(image.shape[1] / abs_scale_width)
        new_height = int(image.shape[0] / abs_scale_height)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    return resized_image



def filter_mask_based_on_distance(mask1, mask2, distance_threshold):
    """
    Filters mask1 by removing pixels that are further than distance_threshold from any pixel in mask2.

    Args:
    mask1 (np.array): Input mask where 255 indicates pixels of interest.
    mask2 (np.array): Reference mask where 255 indicates pixels of interest.
    distance_threshold (float): Maximum distance threshold for pixels in mask1 to be retained.

    Returns:
    np.array: The filtered version of mask1.
    """
    # Convert masks to boolean where True represents the pixels of interest (255 in original mask)
    mask1_bool = mask1 == 255
    mask2_bool = mask2 == 255

    # Compute the distance transform of the inverted mask2
    # distance_transform_edt calculates the Euclidean distance to the nearest zero (False) pixel
    distance_from_mask2 = distance_transform_edt(~mask2_bool)

    # Create a new mask where mask1 pixels are retained only if their distance to the nearest mask2 pixel is within the threshold
    filtered_mask1 = np.where((mask1_bool & (distance_from_mask2 <= distance_threshold)), 255, 0)

    return filtered_mask1.astype(np.uint8)


def keep_n_pixels_in_column(binary_mask, n):
    """
    Keep subsequent N pixels starting from the first encountered 255 pixel in each column using a vectorized approach.

    Args:
    binary_mask (numpy.array): A 2D binary image where the objects are 255 and the background is 0.
    n (int): Number of pixels to keep after the first encountered 255 pixel in each column.

    Returns:
    numpy.array: Modified binary mask with only the specified number of pixels retained in each column.
    """
    # Get the height and width of the image
    height, width = binary_mask.shape

    # Create an output mask initialized to zero
    output_mask = np.zeros_like(binary_mask)

    # Find the first occurrence of 255 in each column using broadcasting
    first_indices = np.argmax(binary_mask == 255, axis=0)

    # Create an array of indices for each column
    row_indices = np.arange(height).reshape(-1, 1)

    # Calculate masks for each column where the first 255 pixel and the next n pixels should be 1
    valid_mask = (row_indices >= first_indices) & (row_indices < first_indices + n)

    # Apply the valid mask to set the corresponding positions in output_mask to 255
    output_mask[valid_mask] = binary_mask[valid_mask]

    return output_mask

def get_contour_of_binary_image(binary):
    """
    Extract contours from a binary image and return an image where the contour pixels are 255 and others are 0.

    Args:
    image_path (str): Path to the binary image file.

    Returns:
    numpy.ndarray: An image with contours drawn.
    """

    # Find contours using OpenCV
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw contours
    contour_img = np.zeros_like(binary)

    # Draw the contours
    cv2.drawContours(contour_img, contours, -1, (255), thickness=1)

    return contour_img


def connect_close_endpoints(binary_mask, connection_dist):
    # Ensure binary mask is binary

    # Apply thinning
    try:
        thinned = cv2.ximgproc.thinning(binary_mask)
    except Exception as e:
        return None


    # Label connected components
    num_labels, labels_im = cv2.connectedComponents(thinned)
    if num_labels==1:
        return binary_mask

    # Detect endpoints and group them by component
    endpoints = {}
    rows, cols = thinned.shape

    # Detect endpoints
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if thinned[i, j] == 255:
                region = thinned[i-1:i+2, j-1:j+2]
                count = np.sum(region == 255)
                if count == 2:  # Endpoint condition
                    if labels_im[i, j] not in endpoints:
                        endpoints[labels_im[i, j]] = []
                    endpoints[labels_im[i, j]].append((i, j))

    # Initialize output image

    # Draw connections between close endpoints of different components
    for label1, pts1 in endpoints.items():
        for label2, pts2 in endpoints.items():
            if label1 != label2:
                for pt1 in pts1:
                    closest_pt = None
                    min_dist = float('inf')
                    for pt2 in pts2:
                        dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_pt = pt2
                    if min_dist < connection_dist:
                        # Draw a white line if close enough
                        cv2.line(binary_mask, (pt1[1], pt1[0]), (closest_pt[1], closest_pt[0]), 255, 2)

    return binary_mask


def dilate_mask(binary_mask,kernel_size=3):
    """
    Dilate a binary mask using a default 3x3 kernel.

    Args:
    binary_mask (numpy.array): A 2D binary image where the objects are 255 and the background is 0.

    Returns:
    numpy.array: The dilated binary mask.
    """
    # Create a default 3x3 kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation to the binary mask
    dilated_mask = cv2.dilate(binary_mask, kernel)

    return dilated_mask


def erode_mask(binary_mask, kernel_size=3, iterations=1):
    """
    Erode a 2D binary mask to make it thinner.

    Parameters:
        binary_mask (numpy.ndarray): A 2D binary mask (values 0 or 1).
        kernel_size (int): Size of the structuring element (kernel). Default is 3.
        iterations (int): Number of erosion iterations. Default is 1.

    Returns:
        numpy.ndarray: The eroded binary mask.
    """
    # Convert binary mask to uint8 if it's not already
    if binary_mask.dtype != np.uint8:
        binary_mask = (binary_mask * 255).astype(np.uint8)

    # Create a structuring element (kernel) for erosion
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Perform erosion
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=iterations)

    # Convert back to binary format (values 0 or 1)
    eroded_mask = (eroded_mask > 0).astype(np.uint8)

    return eroded_mask