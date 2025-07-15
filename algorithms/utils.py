import math
import random
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.ndimage._morphology import binary_fill_holes
from skimage import exposure, measure
from skimage._shared.utils import _supported_float_type
from skimage.filters import gaussian, sobel
from skimage.util import img_as_float


#^ ==========================ESAVA============================================
def color_CLAHE_transformation(image: np.ndarray) -> np.ndarray:
    assert image.shape[-1] == 3, "Input image must have 3 channels (RGB)."
    channels = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    equalized_channels = [clahe.apply(channel) for channel in channels]
    equalized_image = cv2.merge(equalized_channels)
    
    return equalized_image

def maxRegionFilter(image: np.ndarray) -> np.ndarray:
    labels = measure.label(image)
    regions = measure.regionprops(labels)
    cell_region = max(regions, key=lambda x: x.area)
    image[labels != cell_region.label] = 0
    image = binary_fill_holes(image) * 255.0
    return image

def needAjust(image: np.ndarray) -> bool:
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    hv = np.sqrt(np.mean(h) ** 2 + np.mean(v) ** 2)
    return hv < 120

def color_scale_display(img, Shadow = 0, Highlight = 255, Midtones = 1.0) -> np.ndarray:
    if Highlight > 255:
        Highlight = 255
    if Shadow < 0:
        Shadow = 0
    if Shadow >= Highlight:
        Shadow = Highlight - 2
    if Midtones > 9.99:
        Midtones = 9.99
    if Midtones < 0.01:
        Midtones = 0.01

    img = np.array(img, dtype=np.float16)

    Diff = Highlight - Shadow
    img = img - Shadow
    img[img < 0] = 0
    img = (img / Diff) ** (1 / Midtones) * 255
    img[img > 255] = 255

    img = np.array(img, dtype=np.uint8)
    return img

def enhance_cell(image: np.ndarray) -> np.ndarray:
    pLow, pHigh = np.percentile(image, (1, 99))
    img_rescale = exposure.rescale_intensity(image, in_range=(pLow, pHigh))
    return img_rescale

def grab_cut(image: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(np.uint8(image), mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)  # 10
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask2

#^ ==========================ESAVA============================================

def dynamic_inpaint(image, mask, min_radius=1, max_radius=15):
    h, w = image.shape[:2]
    total_pixels = h * w
    mask_pixels = np.count_nonzero(mask)

    mask_ratio = mask_pixels / total_pixels

    inpaint_radius = int(mask_ratio * max_radius)
    inpaint_radius = np.clip(inpaint_radius, min_radius, max_radius)

    result = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return result

def get_ellipse_from_bbox(bbox, padding):
    x_min, y_min, x_max, y_max = bbox
    x_min += padding
    y_min += padding
    x_max -= padding
    y_max -= padding
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    semi_major = 0.5 * width
    semi_minor = 0.5 * height
    angle = math.atan2(y_max - y_min, x_max - x_min)
    return center_x, center_y, semi_major, semi_minor, angle

def Get_boundary_ellipse(center_x, center_y, semi_major, semi_minor, angle, num_keypoint = 300):
    t = np.linspace(0, 2 * np.pi, num_keypoint)
    ellipse_x = (
        center_x
        + semi_major * np.cos(t) * np.cos(angle)
        - semi_minor * np.sin(t) * np.sin(angle)
    )
    ellipse_y = (
        center_y
        + semi_major * np.cos(t) * np.sin(angle)
        + semi_minor * np.sin(t) * np.cos(angle)
    )
    boundary = np.column_stack((ellipse_y, ellipse_x))
    boundary = boundary.astype(np.uint8)
    return boundary

def initialize_rotated_ellipse_boundary(bbox, padding, angle, num_points = 1000):
    x_min, y_min, x_max, y_max = (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    )

    if angle is None:
        angle = math.atan2(y_max - y_min, x_max - x_min)

    x_c = (x_min + x_max) / 2
    y_c = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    semi_major_axis = width / 2
    semi_minor_axis = height / 2
    angle = np.radians(angle)
    
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = np.round(
        x_c
        + semi_major_axis * np.cos(theta) * np.cos(angle)
        - semi_minor_axis * np.sin(theta) * np.sin(angle)
    )
    y = np.round(
        y_c
        + semi_major_axis * np.cos(theta) * np.sin(angle)
        + semi_minor_axis * np.sin(theta) * np.cos(angle)
    )

    return np.column_stack((y, x))

def get_subimage(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image[ymin:ymax, xmin:xmax]

def plot_vector_field(img, axes, vx, vy, coordX, coordY):
    scale = np.sqrt(np.max(vx**2 + vy**2)) * 20.0
    axes.imshow(img, cmap="gray")
    axes.axis("off")

    axes.quiver(
        coordX,
        coordY,
        vx[coordY, coordX],
        -vy[coordY, coordX],
        scale=scale,
        color="blue",
        headwidth=5,
    )

def edge_map(image, sigma):
    blur = gaussian(image, sigma)
    return sobel(blur)

def gradient_field(image):
    im = gaussian(image, 1)
    gradx = np.hstack([im[:, 1:], im[:, -2:-1]]) - np.hstack(
        [im[:, 0:1], im[:, :-1]]
    )
    grady = np.vstack([im[1:, :], im[-2:-1, :]]) - np.vstack(
        [im[0:1, :], im[:-1, :]]
    )
    return gradx, grady  # return sobel x and sobel y

def gradient_field_sobel(image):
    im = gaussian(image, 1)
    gX = cv2.Sobel(im, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gY = cv2.Sobel(im, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    return gX, gY  # return sobel x and sobel y

def gradient_vector_flow(img, fx, fy, mu, dx=1.0, dy=1.0, eps=1.e-6):
    """
    calc gradient vecotr flow of input gradient field fx, fy
    """
    b = fx**2.0 + fy**2.0
    c1, c2 = b * fx, b * fy

    r = 0.25
    
    dt = r * dx * dy / mu  

    N = int(max(1, np.sqrt(img.shape[0] * img.shape[1])))
    curr_u = fx
    curr_v = fy

    def laplacian(m):
        return (
            np.hstack([m[:, 0:1], m[:, :-1]])
            + np.hstack([m[:, 1:], m[:, -2:-1]])
            + np.vstack([m[0:1, :], m[:-1, :]])
            + np.vstack([m[1:, :], m[-2:-1, :]])
            - 4 * m
        )

    for _ in range(N):
        next_u = (1.0 - b * dt) * curr_u + r * laplacian(curr_u) + c1 * dt
        next_v = (1.0 - b * dt) * curr_v + r * laplacian(curr_v) + c2 * dt
        
        delta = np.sqrt((curr_u - next_u) ** 2 + (curr_v - next_v) ** 2)
        if np.mean(delta) < eps:
            break
        else:
            curr_u, curr_v = next_u, next_v

    return curr_u, curr_v

def combine_sobel_maps(sobel_x, sobel_y):
    # Calculate the gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Calculate the gradient direction (in radians)
    gradient_direction = np.arctan2(sobel_y, sobel_x)

    # Normalize the gradient magnitude to the range 0 to 1 (optional)
    gradient_magnitude = np.interp(
        gradient_magnitude,
        (gradient_magnitude.min(), gradient_magnitude.max()),
        (0, 1),
    )

    return gradient_magnitude, gradient_direction


def Get_boundary_circle(center_x, center_y, radius):  # return set of boundary (x, y)
    num_keypoints = 500
    theta_vals = np.linspace(0, 2 * np.pi, num_keypoints, endpoint=False)
    keypoints = []
    for theta in theta_vals:
        x = int(center_x + radius * np.cos(theta))
        y = int(center_y + radius * np.sin(theta))
        keypoints.append(
            (y, x)
        )  # active contour model of the skimage are using (y, x)
    return keypoints

def blur_boundary(image, mask, kernel_size=3):
    """
    Blurs the boundary of a region in the image defined by the mask.

    :param image: Input image (numpy array).
    :param mask: Mask defining the boundary (numpy array, same size as image).
    :param kernel_size: Size of the neighborhood for blurring.
    :return: Image with blurred boundary.
    """
    blurred_image = image.copy()
    rows, cols = image.shape[:2]
    k = kernel_size // 2  # Half kernel size

    # Iterate through boundary pixels
    y_indices, x_indices = np.where(mask == 1)
    for y, x in zip(y_indices, x_indices):
        # Define the neighborhood
        y_min = max(0, y - k)
        y_max = min(rows, y + k + 1)
        x_min = max(0, x - k)
        x_max = min(cols, x + k + 1)

        # Randomly select a pixel from the neighborhood
        rand_y = random.randint(y_min, y_max - 1)
        rand_x = random.randint(x_min, x_max - 1)
        blurred_image[y, x] = image[rand_y, rand_x]

    return blurred_image

def edge_map(image, sigma):
    blur = gaussian(image, sigma)
    return sobel(blur)

def Shoelace(points):
    """
    Calculate the area of a polygon using the Shoelace formula.
    Points should be a list of (x, y) tuples or 2D points.
    """
    x = [p[1] for p in points]
    y = [p[0] for p in points]
    
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def uniformity_evaluation(contours: List[List[Tuple]]):
    if len(contours) == 0:
        return 0
    
    areas = [Shoelace(contour) for contour in contours]
    
    mean_area = np.mean(areas)
    std_area = np.std(areas)

    if mean_area != 0:
        cv = std_area / mean_area
    else:
        cv = 0 
    
    return cv

def plot_vector_field_to_image(img):
    H, W = img.shape[:2]

    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[-1] == 3 else img
    gray_image = img_as_float(gray_image)
    gray_image = gray_image.astype(_supported_float_type(gray_image.dtype), copy=False)

    edge = edge_map(gray_image, sigma=2)
    fx, fy = np.gradient(edge)
    gx, gy = gradient_vector_flow(edge, fx, fy, mu=0.3)

    coordY, coordX = np.meshgrid(np.arange(0, H, 3), np.arange(0, W, 3), indexing="ij")

    annotate_image = (np.copy(img) * 255).astype(np.uint32)
    scale_fx_fy = np.sqrt(np.max(fx**2 + fy**2)) * 20.0
    scale_gx_gy = np.sqrt(np.max(gx**2 + gy**2)) * 20.0

    fig, axes = plt.subplots(1, 2, figsize=(2 * W / 100, H / 100), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas = FigureCanvas(fig)

    axes[0].imshow(annotate_image, cmap="gray")
    axes[0].quiver(coordX, coordY, fx[coordY, coordX], -fy[coordY, coordX], scale=scale_fx_fy, color="green")
    axes[0].axis("off")

    axes[1].imshow(annotate_image, cmap="gray")
    axes[1].quiver(coordX, coordY, gx[coordY, coordX], -gy[coordY, coordX], scale=scale_gx_gy, color="blue")
    axes[1].axis("off")

    canvas.draw()
    raw_width, raw_height = map(int, fig.get_size_inches() * fig.get_dpi())
    image_np = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(raw_height, raw_width, 3)

    plt.close(fig)

    expected_width = W * 2
    image_np = cv2.resize(image_np, (expected_width, H), interpolation=cv2.INTER_LINEAR)

    return image_np
