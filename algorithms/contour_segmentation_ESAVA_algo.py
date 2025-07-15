# -----------------------------------------------------------------------------
# Author: Tan Nguyen <ngduytan288@gmail.com> | <https://github.com/dyytnn>
# Reimplementation of ESAVA: Efficient Structure-Aware Vision Attention
# Original GitHub: https://github.com/fsccycy/ESAVA
# Corresponding Paper:
# Z. Liao et al., "A clinical consensus-compliant deep learning approach to
# quantitatively evaluate human in vitro fertilization early embryonic development
# with optical microscope images," Artif Intell Med., vol. 149, 2024.
# DOI: 10.1016/j.artmed.2024.102773
# -----------------------------------------------------------------------------

from functools import partial

import cv2
import numpy as np
from skimage.restoration import calibrate_denoiser, denoise_tv_chambolle

from abstract import AbstractProcess

from .utils import grab_cut, maxRegionFilter


class OriginalESAVAContourSegmentation(AbstractProcess):
    def __init__(self):
        super().__init__()
        
    def _pre_process(self, data):
        image = data.get("image", None)  # todo: color image 
        box = data.get("bounding_box", None)
        
        return image, (box[0], box[1], box[2] - box[0], box[3] - box[1])
    
    def _main_process(self, data):
        image, rectangle = data
        mask = np.zeros(image.shape[:2], np.uint8)
        canvas = grab_cut(image, rectangle)

        if np.max(canvas) != 0:    
            canvas = maxRegionFilter(canvas)
            contours, _ = cv2.findContours(np.uint8(canvas), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mask = cv2.drawContours(mask, contours, -1, 255, 1)
            
        return mask
    
    def _post_process(self, data):
        return data
    
    
class LocalPatchProcessESAVAContourSegmentation(AbstractProcess):
    """
        This process re-implements ESAVA segmentation for each individual blastomere.
        It enhances each blastomere's local patch (e.g., denoising or contrast) before applying GrabCut-based contour segmentation.
    """
    def __init__(self):
        super().__init__()
        
    def _pre_process(self, data):
        """
            Inputs:
            - image: original RGB image
            - enhanced_local_patch: pre-enhanced patch of a single blastomere
            - bounding_box: the [x1, y1, x2, y2] of the blastomere
            - padded_bounding_box: extended box used for patch embedding

            Returns:
            - patched image with enhanced patch replaced in context
            - rectangle for GrabCut in (x, y, w, h) format
        """
        image = data.get("image", None)  # todo: color image 
        image_to_enhance = np.copy(image)
        enhanced_local_patch = data.get("enhanced_local_patch", None)
        
        box = data.get("bounding_box", None)
        padded_box = data.get("paddded_bounding_box", None)
        
        image_to_enhance[padded_box[1] : padded_box[3], padded_box[0] : padded_box[2], ...] = enhanced_local_patch
        
        return image_to_enhance, (box[0], box[1], box[2] - box[0], box[3] - box[1])
    
    def _main_process(self, data):
        """
            Applies GrabCut + contour extraction on locally enhanced patch
        """
        image_to_enhance, rectangle = data
        
        mask = np.zeros(image_to_enhance.shape[:2], np.uint8)
        canvas = grab_cut(image_to_enhance, rectangle)

        if np.max(canvas) != 0:    
            canvas = maxRegionFilter(canvas)
            contours, _ = cv2.findContours(np.uint8(canvas), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mask = cv2.drawContours(mask, contours, -1, 255, 1)
            
        return mask
            
    def _post_process(self, data):
        return data
    
    def process(self, data):
        return super().process(data)
    