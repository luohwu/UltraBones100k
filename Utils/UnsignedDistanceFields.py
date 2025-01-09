import numpy as np
import skimage.morphology
from scipy.ndimage import binary_fill_holes as nd_binary_fill_holes, distance_transform_edt
from skimage.morphology import closing, square,disk,erosion,skeletonize
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import cv2
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def compute_unsigned_distance_2Dmap(binary_image,truncated=200,skeletonize=False):
    """
    Compute the distance map for a label image where each pixel's value
    is the distance to the closest white (value 255) pixel.

    Args:
    label_image (numpy.ndarray): A label image where white pixels (255) represent areas of interest.

    Returns:
    numpy.ndarray: Distance map where each pixel's value is the distance to the nearest white pixel (255).
    """
    # Ensure the image is boolean where True (1) is 255 and False (0) is the background.
    binary_image = binary_image > 0

    if skeletonize:
        binary_image=skimage.morphology.skeletonize(binary_image).astype(np.uint8)*255
        kernel_size = 3  # Diameter of the circle
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # binary_image=cv2.dilate(binary_image,kernel,iterations=1)
        # cv2.imshow("skeleton",binary_image);        cv2.waitKey(0)


    # Compute the distance transform: distance from non-object (0) pixels to the nearest object (1) pixel
    distance_map = distance_transform_edt(~binary_image)
    if truncated>0:
        distance_map[distance_map > truncated] = truncated
    # Create the signed distance field where inside mask distances are negative
    return distance_map


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

def visualize_unsigned_distance_map(image, label, distance_map):
    """
    Visualize an image, a label, and a distance map in three subplots and display using OpenCV.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Label image
    axes[1].imshow(label, cmap='gray')
    axes[1].set_title('Label Image')
    axes[1].axis('off')

    # Distance map
    cmap = plt.cm.viridis
    norm = Normalize(vmin=np.min(distance_map), vmax=np.max(distance_map))
    im = axes[2].imshow(distance_map, cmap=cmap, norm=norm)
    axes[2].set_title('Distance Map')
    axes[2].axis('off')

    # Colorbar for the distance map
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Distance Values')

    # Convert figure to OpenCV image format
    opencv_image = plt_to_opencv(fig)
    plt.close(fig)
    return opencv_image


    # Close the figure to free memory
