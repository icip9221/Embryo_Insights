from functools import partial

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
from skimage import filters, morphology, restoration
from skimage.feature import canny
from skimage.filters import gaussian, sobel
from skimage.morphology import closing, dilation, disk, erosion, opening
from skimage.restoration import calibrate_denoiser, denoise_tv_chambolle

from abstract import AbstractProcess

from .utils import color_scale_display, enhance_cell, needAjust


def ESAVA_denoising(local_patch: np.ndarray) -> np.ndarray:
    """
        Apply denoising to a local blastomere patch following ESAVA's approach.

        Steps:
        1. Optionally adjust contrast via color scaling if `needAjust()` returns True.
        2. Calibrate the best denoising parameter using `calibrate_denoiser` with Total Variation (TV) denoising.
        3. Enhance the result using `enhance_cell()` for better contrast and visibility.

        Parameters:
        - local_patch (np.ndarray): A cropped image patch (blastomere) in grayscale or RGB.

        Returns:
        - np.ndarray: Enhanced patch after denoising and contrast adjustment.
    """
    if needAjust(local_patch):
        local_patch = color_scale_display(local_patch, 0, 170, 1.53)
    
    _denoise_tv = partial(denoise_tv_chambolle, channel_axis=None)
    calibrated_denoiser = calibrate_denoiser(
        local_patch,
        _denoise_tv,
        denoise_parameters={"weight": np.arange(0.1, 0.9, 0.1)}
    )
    local_patch_enhanced = enhance_cell(np.uint8(calibrated_denoiser(local_patch) * 255))
    
    return local_patch_enhanced
        

def _kernel_denoising(image: np.ndarray, **kwargs) -> np.ndarray:
    """
        Apply various kernel-based denoising methods for blastomere images.

        Supported Methods:
        - 'bilateral': Bilateral filter (edge-preserving)
        - 'non_local_means': Fast Non-local Means denoising
        - 'gaussian': Gaussian smoothing from skimage
        - 'wiener': Wiener filter from skimage.restoration
        - 'tv_denoise': Total Variation denoising (TV-Chambolle)
        - 'adaptive': Bilateral filter + Gaussian (hybrid method)

        Parameters:
        - image (np.ndarray): Input grayscale or RGB image.
        - method (str): Denoising method to use (default: 'gaussian').
        - denoiser_params (dict): Additional parameters passed to the denoising function.

        Returns:
        - np.ndarray: Denoised image as float32 in range [0, 1].
    """
    method = kwargs.get("method", "gaussian")
    params = kwargs.get("denoiser_params", {})
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    if method == 'bilateral':
        denoised = cv2.bilateralFilter(
            (image * 255).astype(np.uint8), 
            d=9, sigmaColor=75, sigmaSpace=75, **params
        ).astype(np.float32) / 255.0
        
    elif method == 'non_local_means':
        denoised = cv2.fastNlMeansDenoising(
            (image * 255).astype(np.uint8),
            None, 10, 7, 21, **params
        ).astype(np.float32) / 255.0
        
    elif method == 'gaussian':
        denoised = gaussian(image, **params, preserve_range=False)
        
    elif method == 'wiener':
        denoised = restoration.wiener(image, noise=0.1, **params)
        
    elif method == 'tv_denoise':
        denoised = restoration.denoise_tv_chambolle(image, weight=0.1, **params)
        
    elif method == 'adaptive':
        bilateral = cv2.bilateralFilter(
            (image * 255).astype(np.uint8), 
            d=9, sigmaColor=75, sigmaSpace=75, **params
        ).astype(np.float32) / 255.0
        
        denoised = gaussian(bilateral, sigma=0.5, preserve_range=False)
        
    return denoised

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance the contrast of a grayscale image using CLAHE and gamma correction.

    Steps:
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances local contrast.
    2. Gamma correction further adjusts intensity non-linearly to improve edge visibility.

    Parameters:
    - image (np.ndarray): Grayscale image in [0, 1] or [0, 255].

    Returns:
    - np.ndarray: Enhanced image in float32, normalized to [0, 1].
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply((image * 255).astype(np.uint8))
    
    # Gamma correction
    gamma = 1.2
    enhanced = np.power(enhanced.astype(np.float32) / 255.0, gamma)
    
    return enhanced

def _structure_preserving_denoise(image: np.ndarray) -> np.ndarray:
    """
        Apply structure-preserving denoising on a grayscale image.

        This method:
        - Detects edge structures using both Canny and Sobel filters.
        - Applies TV denoising to smooth non-edge regions.
        - Applies bilateral filtering to preserve edge-aware regions.
        - Finishes with a light bilateral smoothing across the entire image for final refinement.

        Parameters:
        - image (np.ndarray): A grayscale image normalized in [0, 1].

        Returns:
        - np.ndarray: Denoised image with structure preserved, in float32 format [0, 1].
    """
    edges_canny = canny(image, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
    edges_sobel = filters.sobel(image) > 0.1
    edges_combined = np.logical_or(edges_canny, edges_sobel)
    
    structure_mask = dilation(edges_combined, disk(3))

    non_edge_mask = ~structure_mask
    denoised = image.copy()

    internal_denoised = restoration.denoise_tv_chambolle(image, weight=0.15)
    denoised[non_edge_mask] = internal_denoised[non_edge_mask]
    
    edge_denoised = restoration.denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15)
    denoised[structure_mask] = edge_denoised[structure_mask]
    
    denoised = restoration.denoise_bilateral(denoised, sigma_color=0.03, sigma_spatial=10)
    
    return denoised

def _anisotropic_diffusion(image: np.ndarray, num_iter=20, delta_t=0.1, kappa=0.1) -> np.ndarray:
    """
        Apply Perona-Malik anisotropic diffusion for edge-preserving denoising.

        This method reduces noise while preserving edges by adapting the smoothing
        strength based on local gradient magnitude.

        Parameters:
        - image (np.ndarray): Grayscale image, ideally normalized in [0, 1]
        - num_iter (int): Number of diffusion iterations (default: 20)
        - delta_t (float): Integration constant, should be <= 0.25 for stability (default: 0.1)
        - kappa (float): Controls edge sensitivity. Lower values preserve edges more (default: 0.1)

        Returns:
        - np.ndarray: Diffused (denoised) image, same shape as input
    """
    img = image.copy()
    
    for i in range(num_iter):        
        grad_n = np.roll(img, -1, axis=0) - img  
        grad_s = np.roll(img, 1, axis=0) - img   
        grad_e = np.roll(img, -1, axis=1) - img  
        grad_w = np.roll(img, 1, axis=1) - img   
        
        c_n = np.exp(-(grad_n/kappa)**2)
        c_s = np.exp(-(grad_s/kappa)**2)
        c_e = np.exp(-(grad_e/kappa)**2)
        c_w = np.exp(-(grad_w/kappa)**2)
        
        img += delta_t * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
    
    return img

def _guided_filter_denoise(image: np.ndarray, radius=8, epsilon=0.01) -> np.ndarray:
    """
    Apply guided filtering for edge-preserving smoothing.

    Guided filtering uses a guidance image (here the image itself) to smooth 
    the input while preserving strong edges.

    Parameters:
    - image (np.ndarray): Grayscale image in [0, 1] range.
    - radius (int): Radius of the local window used for filtering.
    - epsilon (float): Regularization parameter to control edge sensitivity.

    Returns:
    - np.ndarray: Smoothed image in float64 format, range [0, 1].
    """
    def guided_filter(I, p, r, eps):
        mean_I = uniform_filter(I, size=r)
        mean_p = uniform_filter(p, size=r)
        mean_Ip = uniform_filter(I * p, size=r)
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = uniform_filter(I * I, size=r)
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = uniform_filter(a, size=r)
        mean_b = uniform_filter(b, size=r)
        
        return mean_a * I + mean_b
    
    return guided_filter(image, image, radius, epsilon)

def _adaptive_hybrid_denoise(image: np.ndarray) -> np.ndarray:
    """
    Adaptive hybrid denoising using edge-aware selective filtering.

    This method:
    - Detects edge zones using Canny and Sobel edge detectors.
    - Computes distance from edges to define internal vs boundary regions.
    - Applies Non-Local Means denoising to inner regions.
    - Applies Bilateral filtering to edge regions.
    - Finishes with a guided filter pass to blend all regions smoothly.

    Parameters:
    - image (np.ndarray): Grayscale image in [0, 1] range.

    Returns:
    - np.ndarray: Enhanced and denoised image in float64 format.
    """
    from scipy.ndimage import distance_transform_edt
    
    edges_canny = canny(image, sigma=1.5, low_threshold=0.08, high_threshold=0.15)
    edges_sobel = filters.sobel(image) > filters.threshold_otsu(filters.sobel(image))
    
    edges_combined = np.logical_or(edges_canny, edges_sobel)
    
    edge_distance = distance_transform_edt(~edges_combined)
    
    edge_distance = edge_distance / edge_distance.max()
        
    denoised = image.copy()
        
    internal_mask = edge_distance > 0.3
    if np.any(internal_mask):        
        img_uint8 = (image * 255).astype(np.uint8)
        nlm_denoised = cv2.fastNlMeansDenoising(img_uint8, None, 12, 7, 21)
        denoised[internal_mask] = (nlm_denoised[internal_mask].astype(np.float64) / 255.0)
        
    edge_mask = edge_distance <= 0.3
    if np.any(edge_mask):        
        img_uint8 = (image * 255).astype(np.uint8)
        bilateral_denoised = cv2.bilateralFilter(img_uint8, 9, 40, 40)
        denoised[edge_mask] = (bilateral_denoised[edge_mask].astype(np.float64) / 255.0)
        
    denoised = _guided_filter_denoise(denoised, radius=5, epsilon=0.005)
    
    return denoised

def enhance_boundary_contrast(image: np.ndarray, method='adaptive_clahe') -> np.ndarray:
    """
    Enhance boundary contrast of a grayscale image using different strategies.

    Options:
    - 'adaptive_clahe': CLAHE followed by gamma boost at boundary regions.
    - 'unsharp_masking': Enhance edges by subtracting Gaussian blur.
    - 'morphological_gradient': Enhance transitions using dilation-erosion gradient.

    Parameters:
    - image (np.ndarray): Grayscale image in [0, 1].
    - method (str): Enhancement method to apply.

    Returns:
    - np.ndarray: Contrast-enhanced image in [0, 1].
    """
    
    if method == 'adaptive_clahe':
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply((image * 255).astype(np.uint8))

        edges = canny(image, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
        boundary_mask = dilation(edges, disk(2))
    
        enhanced_float = enhanced.astype(np.float64) / 255.0
        gamma_enhanced = np.power(enhanced_float, 0.8) 
        
        result = enhanced_float.copy()
        result[boundary_mask] = gamma_enhanced[boundary_mask]
        
        return result
        
    elif method == 'unsharp_masking':
        blurred = gaussian(image, sigma=2.0)
        unsharp = image + 0.5 * (image - blurred)
        return np.clip(unsharp, 0, 1)
        
    elif method == 'morphological_gradient':
        se = disk(1)
        gradient = morphology.dilation(image, se) - morphology.erosion(image, se)
        enhanced = image + 0.3 * gradient
        return np.clip(enhanced, 0, 1)

def create_boundary_focused_edge_map(image: np.ndarray) -> np.ndarray:
    """
    Generate a boundary-focused edge map using multi-scale gradient fusion.

    Combines edge maps from multiple Gaussian-smoothed versions of the image, 
    weighted by scale, to suppress internal noise and emphasize clear boundaries.

    Parameters:
    - image (np.ndarray): Grayscale image in [0, 1].

    Returns:
    - np.ndarray: Binary edge map in float64 format (values 0 or 1).
    """
    edges_final = np.zeros_like(image)
    scales = [1.0, 1.5, 2.0, 2.5]
    weights = [0.4, 0.3, 0.2, 0.1]
    
    for scale, weight in zip(scales, weights):        
        smoothed = gaussian(image, sigma=scale)
                
        grad_y, grad_x = np.gradient(smoothed)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
        threshold = np.percentile(magnitude, 85)
        strong_edges = magnitude > threshold
                
        edges_final += weight * strong_edges.astype(np.float64)
        
    edges_final = edges_final / edges_final.max()
        
    edges_final = closing(edges_final > 0.3, disk(1))
    edges_final = opening(edges_final, disk(1))
    
    return edges_final.astype(np.float64)

def _edge_preserving_bilateral(image: np.ndarray) -> np.ndarray:
    """
    Perform multi-pass bilateral filtering for edge-preserving smoothing.

    Applies bilateral filtering with progressively decreasing kernel sizes 
    and strength to remove noise while keeping edges sharp.

    Parameters:
    - image (np.ndarray): Grayscale image in [0, 1].

    Returns:
    - np.ndarray: Smoothed image in float64 format, range [0, 1].
    """
    img_uint8 = (image * 255).astype(np.uint8)
    denoised = cv2.bilateralFilter(img_uint8, 15, 80, 80)

    denoised = cv2.bilateralFilter(denoised, 9, 50, 50)

    denoised = cv2.bilateralFilter(denoised, 5, 30, 30)
    
    return denoised.astype(np.float64) / 255.0

class MicroscopicBlastomereEnhancer(AbstractProcess):
    """
    A configurable enhancement module for denoising microscopic blastomere cell patches.

    This class provides a unified interface for applying various denoising and enhancement
    techniques to localized regions of microscopic embryo images, particularly blastomere
    cells, to improve edge clarity and structural fidelity for downstream segmentation.

    Supported methods:
    - 'structure_preserving': Total variation + bilateral denoising with edge masks.
    - 'edge_preserving_bilateral': Multi-pass bilateral filtering.
    - 'anisotropic_diffusion': Perona-Malik diffusion preserving edges.
    - 'guided_filter': Guided filter-based edge-aware smoothing.
    - 'adaptive_hybrid': Region-based fusion of bilateral and non-local means denoising.
    - 'kernel_denoising': Traditional filters (Gaussian, bilateral, TV, etc.).
    - 'local_patch_esava': Reimplementation of ESAVA-specific local patch enhancement.

    Parameters:
    - config (dict): Configuration dictionary containing:
        - 'method': Name of the denoising method to apply.
        - 'denoiser_params': (Optional) Additional parameters for the denoiser.

    Expected Input (`data` dict):
    - 'image': Full image (in range 0–255, RGB or grayscale).
    - 'boxes': A bounding box in [x1, y1, x2, y2] format indicating the local patch to process.

    Returns:
    - The enhanced local patch, denoised using the selected method.
    - If `method` is 'local_patch_esava', padding is added before denoising.

    Notes:
    - Non-ESAVA methods convert images to grayscale and normalize to [0, 1].
    - All enhancements are applied to a cropped local region defined by the bounding box.
    - Post-processing currently returns the raw denoised patch.

    Example usage:
        config = {"method": "anisotropic_diffusion", "denoiser_params": {"num_iter": 10, "kappa": 0.1}}
        enhancer = MicroscopicBlastomereEnhancer(config)
        output = enhancer.process({"image": image, "boxes": [x1, y1, x2, y2]})
    """
    def __init__(self, config: dict):
        self.config = config
    
    @property
    def denoiser(self):
        method = self.config.get("method", "structure_preserving")
        if method == 'structure_preserving':
            print("[INFO] _structure_preserving denoising ...")
            return _structure_preserving_denoise
            
        elif method == 'edge_preserving_bilateral':
            print("[INFO] _edge_preserving_bilateral denoising ...")
            return _edge_preserving_bilateral
            
        elif method == 'anisotropic_diffusion':        
            print("[INFO] _anisotropic_diffusion denoising ...")
            return _anisotropic_diffusion
            
        elif method == 'guided_filter':        
            print("[INFO] _guided_filte denoising ...")
            return _guided_filter_denoise
            
        elif method == 'adaptive_hybrid':        
            print("[INFO] _adaptive_hybrid denoising ...")
            return _adaptive_hybrid_denoise
        
        elif method == "kernel_denoising":        
            print("[INFO] _kernel denoising ...")
            return _kernel_denoising

        elif method == "local_patch_esava":
            print("[INFO] ESAVA denoising ...")
            return ESAVA_denoising
    
    def _pre_process(self, data: dict):
        image: np.ndarray = data.get("image", None) #todo image in the 0-255 range
        preprocessing_image = np.copy(image)
        bounding_box = data.get("boxes", None)

        if self.config.get("method", "structure_preserving") == "local_patch_esava":
            padding = 10
            x1, y1, x2, y2 = bounding_box 
            padded_x1 = x1 - padding
            padded_y1 = y1 - padding
            padded_x2 = x2 + padding
            padded_y2 = y2 + padding
            
            return image, [padded_x1, padded_y1, padded_x2, padded_y2]
        
        preprocessing_image = cv2.cvtColor(preprocessing_image, cv2.COLOR_BGR2GRAY) if preprocessing_image.shape[-1] == 3 else preprocessing_image
        preprocessing_image = preprocessing_image.astype(np.float32) / 255.0
        
        return preprocessing_image, bounding_box
    
    def _main_process(self, data):
        preprocessing_image, bounding_box = data
        
        local_patch = preprocessing_image[bounding_box[1] : bounding_box[3], bounding_box[0] : bounding_box[2]]
        
        denoised_local_patch = self.denoiser(local_patch, **self.config.get("denoiser_params")) if self.config.get("denoiser_params") else self.denoiser(local_patch)
        
        denoised_local_patch = denoised_local_patch if self.config.get("method", None) == "local_patch_esava" else np.clip(denoised_local_patch, 0.0, 1.0)
    
        return denoised_local_patch
            
    def _post_process(self, data):
        return data
    