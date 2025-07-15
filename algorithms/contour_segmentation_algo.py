# -----------------------------------------------------------------------------
# Author: Tan Nguyen <ngduytan288@gmail.com> | <https://github.com/dyytnn>
# Date: 2025-10-07
# Description: Gradient Vector Flow implementation and Snake Active Contour model applied in blastomere's cell boundary recognization. 
# -----------------------------------------------------------------------------


import random
import sys

import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage import morphology
from skimage._shared.utils import _supported_float_type
from skimage.filters import sobel
from skimage.util import img_as_float
from tqdm import trange

from abstract import AbsAlgorithmProcessHandler, AbstractProcess
from registry import Blastomere_evaluation_process

from .blastomere_cell_enhancing_algo import (
    ESAVA_denoising,
    MicroscopicBlastomereEnhancer,
)
from .contour_segmentation_ESAVA_algo import (
    LocalPatchProcessESAVAContourSegmentation,
    OriginalESAVAContourSegmentation,
)
from .utils import (
    dynamic_inpaint,
    gradient_field_sobel,
    gradient_vector_flow,
    initialize_rotated_ellipse_boundary,
    plot_vector_field_to_image,
    sobel,
)


class SnakeActiveContour(AbstractProcess):
    def __init__(self, config: dict):
        super(SnakeActiveContour, self).__init__()
        self.config = config
        self.sfixed = False if self.config.get("boundary_condition", "periodic").startswith("fixed") else False
        self.efixed = True if self.config.get("boundary_condition", "periodic").endswith("fixed") else False
        self.sfree = True if self.config.get("boundary_condition", "periodic").startswith("free") else False
        self.efree = True if self.config.get("boundary_condition", "periodic").endswith("free") else False
        
    @property
    def tension(self):
        """
        Compute the tension matrix used in the snake model for contour refinement.

        The tension matrix is derived from an identity matrix by applying cyclic shifts
        to capture the first-order differences between neighboring points in the contour.
        This matrix contributes to the smoothness term in the snake energy, which controls
        the tension or elasticity of the contour.

        Returns:
            np.ndarray: The tension matrix of shape (length_coordinate_points, length_coordinate_points).
        """

        eye_n = np.eye(self.config.get("length_coordinate_points"), dtype=float)
        return (np.roll(eye_n, -1, axis=0) + np.roll(eye_n, -1, axis=1) - 2 * eye_n) 
    
    @property
    def rigidity(self):
        """
        Compute the rigidity matrix used in the snake model for contour refinement.

        The rigidity matrix is constructed by applying cyclic shifts to an identity matrix
        to capture second-order differences between neighboring points in the contour. This
        matrix contributes to the bending energy term in the snake energy, which controls
        the rigidity or stiffness of the contour.

        Returns:
            np.ndarray: The rigidity matrix of shape (length_coordinate_points, length_coordinate_points).
        """
        eye_n = np.eye(self.config.get("length_coordinate_points"), dtype=float)
        return (
            np.roll(eye_n, -2, axis=0)
            + np.roll(eye_n, -2, axis=1)
            - 4 * np.roll(eye_n, -1, axis=0)
            - 4 * np.roll(eye_n, -1, axis=1)
            + 6 * eye_n
        )
        
    @property
    def RegularizationMatrix(self):
        """
        Compute the regularization matrix for the snake model used in contour refinement.

        The regularization matrix combines the tension and rigidity matrices scaled by
        alpha and beta coefficients, respectively. Depending on the boundary conditions
        (fixed or free), specific adjustments are made to the matrix to enforce the
        desired boundary behavior.

        Returns:
            np.ndarray: The regularization matrix of shape 
            (length_coordinate_points, length_coordinate_points), adjusted for boundary conditions.
        """

        alpha = self.config.get("alpha")
        beta = self.config.get("beta")
        
        regularization_matrix = -alpha * self.tension + beta * self.rigidity
        
        if self.config.get("boundary_condition", "periodic").startswith("fixed"):
            regularization_matrix[0, :] = 0
            regularization_matrix[1, :] = 0
            regularization_matrix[1, :3] = [1, -2, 1]
        
        if self.config.get("boundary_condition", "periodic").endswith("fixed"):
            regularization_matrix[-1, :] = 0
            regularization_matrix[-2, :] = 0
            regularization_matrix[-2, -3:] = [1, -2, 1]
        
        if self.config.get("boundary_condition", "periodic").startswith("free"):
            regularization_matrix[0, :] = 0
            regularization_matrix[0, :3] = [1, -2, 1]
            regularization_matrix[1, :] = 0
            regularization_matrix[1, :4] = [-1, 3, -3, 1]
            
        if self.config.get("boundary_condition", "periodic").endswith("free"):
            regularization_matrix[-1, :] = 0
            regularization_matrix[-1, -3:] = [1, -2, 1]
            regularization_matrix[-2, :] = 0
            regularization_matrix[-2, -4:] = [-1, 3, -3, 1]
        
        return regularization_matrix
    
    @property
    def InvRegularizationMatrix(self):
        """
        Compute the inverse of the regularization matrix for the snake model.

        The inverse regularization matrix is needed to solve the linear system in the
        snake model. The regularization matrix is augmented with a diagonal matrix
        scaled by the gamma coefficient, which controls the balance between the
        regularization and data terms in the snake energy.

        Returns:
            np.ndarray: The inverse regularization matrix of shape 
            (length_coordinate_points, length_coordinate_points).
        """
        eye_n = np.eye(self.config.get("length_coordinate_points"), dtype=float)
        return np.linalg.inv(self.RegularizationMatrix + self.config.get("gamma") * eye_n)
    
    def _pre_process(self, data: dict):
        """
        Preprocess the input data for the snake model.

        This method takes the input image and snake, and computes the edge map,
        line integral, and interpolation of the image. The edge map is computed
        using the Sobel operator, and the line integral is computed by summing
        the pixel values along the snake curve. The interpolation of the image
        is computed using a bivariate spline interpolation.

        Parameters:
            data (dict): A dictionary containing the input image and snake.

        Returns:
            tuple: A tuple containing the snake, interpolation of the image,
            and the data type of the image.
        """
        image: np.ndarray = data.get("image", None)
        snake: np.ndarray = data.get("snake", None)
        
        img = img_as_float(image)
        float_dtype = _supported_float_type(image.dtype)
        img = img.astype(float_dtype, copy=False)

        RGB = img.ndim == 3
        
        if self.config.get("w_edge") != 0:
            if RGB:
                edge = [sobel(img[:, :, 0]), sobel(img[:, :, 1]), sobel(img[:, :, 2])]
            else:
                edge = [sobel(img)]
        else:
            edge = [0]

        if RGB:
            img = self.config.get("w_line") * np.sum(img, axis=2) + self.config.get("w_edge") * sum(edge)
        else:
            img = self.config.get("w_line") * img + self.config.get("w_edge") * edge[0]
        
        intp = RectBivariateSpline(
            np.arange(img.shape[1]), np.arange(img.shape[0]), img.T, kx=2, ky=2, s=0
        )
        return snake, intp, float_dtype
    
    def _main_process(self, data):
        """
        Main process of the snake model.

        This method implements the main process of the snake model, which is
        an iterative process that updates the snake curve based on the
        regularization energy and the data term. The regularization energy is
        computed using the inverse of the regularization matrix, and the data
        term is computed using the line integral of the image. The iterative
        process is terminated when the snake converges or when the maximum
        number of iterations is reached.

        Parameters:
            data (tuple): A tuple containing the snake, interpolation of the
                image, and the data type of the image.

        Returns:
            np.ndarray: The final snake curve of shape (length_coordinate_points, 2).
        """
        snake, intp, float_dtype = data
        snake_xy = snake[:, ::-1]
        
        x = snake_xy[:, 0].astype(float_dtype)
        y = snake_xy[:, 1].astype(float_dtype)
        
        n = len(x)
        xsave = np.empty((self.config.get("convergence_order"), n), dtype=float_dtype)
        ysave = np.empty((self.config.get("convergence_order"), n), dtype=float_dtype)
        
        for idx in trange(self.config.get("max_iterations"), desc="Snake Iteration"):            
            fx = intp(x, y, dx = 1, grid = False).astype(float_dtype, copy=False)  
            fy = intp(x, y, dy = 1, grid = False).astype(float_dtype, copy=False)  

            if self.sfixed:
                fx[0] = 0
                fy[0] = 0
            if self.efixed:
                fx[-1] = 0
                fy[-1] = 0
            if self.sfree:
                fx[0] *= 2
                fy[0] *= 2
            if self.efree:
                fx[-1] *= 2
                fy[-1] *= 2
            xn = self.InvRegularizationMatrix @ (self.config.get("gamma") * x + fx)
            yn = self.InvRegularizationMatrix @ (self.config.get("gamma") * y + fy)

            dx = self.config.get("max_px_move") * np.tanh(xn - x)
            dy = self.config.get("max_px_move") * np.tanh(yn - y)
            
            if self.sfixed:
                dx[0] = 0
                dy[0] = 0
            if self.efixed:
                dx[-1] = 0
                dy[-1] = 0
            x += dx
            y += dy
            
            j = idx % (self.config.get("convergence_order") + 1)
            if j < self.config.get("convergence_order"):
                xsave[j, :] = x
                ysave[j, :] = y
            else:
                dist = np.min(
                    np.max(np.abs(xsave - x[None, :]) + np.abs(ysave - y[None, :]), 1)
                )
                
                if idx % 5 == 0:
                    print(f"Iter {idx} | dist: {dist:.4f} | dx max: {np.max(np.abs(dx))}, dy max: {np.max(np.abs(dy))}", flush=True)

                if dist < self.config.get("convergence"):
                    print(f'Convergence at iterate {idx}')
                    break
            
        return np.stack([y, x], axis=1)
    
    def _post_process(self, data):
        return data

class GVFSnakeActiveContour(SnakeActiveContour):
    def __init__(self, config: dict):
        """
        Initialize a GVF snake model.

        Args:
            config (dict): A dictionary containing the configuration of the model.
                The dictionary should contain the following keys:
                    - `mu` (float): The regularization parameter of the snake model.
        """
        super(GVFSnakeActiveContour, self).__init__(config = config)
 
    def _pre_process(self, data: dict):
        """
        Preprocess the input data for the GVF snake model.

        Args:
            data (dict): A dictionary containing the image and snake data.

        Returns:
            tuple: A tuple containing the snake, intp_gx, and intp_gy arrays.
        """
        image: np.ndarray = data.get("image", None)
        snake: np.ndarray = data.get("snake", None)
        
        img = img_as_float(image)
        float_dtype = _supported_float_type(image.dtype)
        img = img.astype(float_dtype, copy=False)

        edge = sobel(img)
        edge = np.power(edge, 0.8)
        fx, fy = gradient_field_sobel(edge)
        gx, gy = gradient_vector_flow(edge, fx, fy, mu = self.config.get("mu" ,0.2))
        
        intp_gx = RectBivariateSpline(np.arange(gx.shape[1]), np.arange(gx.shape[0]), gx.T, kx=2, ky=2)
        intp_gy = RectBivariateSpline(np.arange(gy.shape[1]), np.arange(gy.shape[0]), gy.T, kx=2, ky=2)

        return snake, intp_gx, intp_gy

    def _main_process(self, data):
        """
        Main process of the GVF snake model.

        Args:
            data (tuple): A tuple containing the snake, intp_gx, and intp_gy arrays.

        Returns:
            np.ndarray: The converged snake points.
        """
        
        snake, intp_gx, intp_gy = data
        snake_xy = snake[:, ::-1]
        
        x = snake_xy[:, 0].astype(np.float32)
        y = snake_xy[:, 1].astype(np.float32)
        
        n = len(x)
        xsave = np.empty((self.config.get("convergence_order"), n), dtype=np.float32)
        ysave = np.empty((self.config.get("convergence_order"), n), dtype=np.float32)
        
        for idx in trange(self.config.get("max_iterations"), desc="Snake Iteration"):            
            fx = intp_gx(x, y, grid=False).astype(np.float32)
            fy = intp_gy(x, y, grid=False).astype(np.float32)

            xn = self.InvRegularizationMatrix @ (self.config.get("gamma") * x + fx)
            yn = self.InvRegularizationMatrix @ (self.config.get("gamma") * y + fy)

            dx = self.config.get("max_px_move") * np.tanh(xn - x)
            dy = self.config.get("max_px_move") * np.tanh(yn - y)

            x += dx
            y += dy

            j = idx % (self.config.get("convergence_order") + 1)
            if j < self.config.get("convergence_order"):
                xsave[j, :] = x
                ysave[j, :] = y
            else:
                dist = np.min(
                    np.max(np.abs(xsave - x[None, :]) + np.abs(ysave - y[None, :]), 1)
                )

                if idx % 5 == 0:
                    print(f"Iter {idx} | dist: {dist:.4f} | dx max: {np.max(np.abs(dx))}, dy max: {np.max(np.abs(dy))}", flush=True)

                if dist < self.config.get("convergence"):
                    print(f'convergence at iterate {idx}')
                    break
            
        return np.stack([y, x], axis=1)
    
    def _post_process(self, data):
        return data
    
@Blastomere_evaluation_process.register("SnakeContourAnalysis")
class SnakeContourAnalysis(AbsAlgorithmProcessHandler):
    def __init__(self, config: dict, type: str = "GVF"):
        """
        Initialize the SnakeContourAnalysis object.

        Parameters
        ----------
        config : dict
            Configuration for the contour analysis.
        type : str, optional
            Type of contour analysis, by default "GVF". Options are "GVF", "Snake", "OriginalESAVA", "LocalPatchESAVA".

        Notes
        -----
        If active_contour_params or enhancer_config is not None in the config, the respective objects are created with the given parameters.
        Otherwise, the default parameters are used.
        """
        super().__init__()
        self.config = config
        model_class_name = {
            "GVF": "GVFSnakeActiveContour",
            "Snake": "SnakeActiveContour",
            "OriginalESAVA": "OriginalESAVAContourSegmentation",
            "LocalPatchESAVA": "LocalPatchProcessESAVAContourSegmentation"
        }[type]

        self.active_contour = getattr(sys.modules[__name__], model_class_name)(**config.get("active_contour_params")) if config.get("active_contour_params") is not None else  getattr(sys.modules[__name__], model_class_name)()
        self.enhancer = MicroscopicBlastomereEnhancer(**self.config.get("enhancer_config")) if self.config.get("enhancer_config") is not None else None 
        self.type = type
                
    def _pre_process(self, data):
        """
        Pre-process the input data based on the type of contour analysis.

        Parameters
        ----------
        data : dict
            Input data containing the image and bounding boxes.

        Returns
        -------
        tuple
            Depending on the type of contour analysis, returns:
            - For 'OriginalESAVA': Processed image with enhanced local patch and the bounding boxes.
            - For 'LocalPatchESAVA': Original image and the bounding boxes.
            - For other types: Grayscale CLAHE-applied image and the bounding boxes.

        Notes
        -----
        - For 'OriginalESAVA', a bias is added to the bounding boxes to calculate a local patch, which is then enhanced using ESAVA denoising.
        - For other types, CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to enhance the contrast of the image.
        """

        image = data.get("image", None)
        boxes = data.get("boxes", None).astype(np.int32)
        
        if self.type == "OriginalESAVA":
            bias = 15
            x1 = np.maximum(min(boxes[:, 0]) - bias, 0)
            y1 = np.maximum(min(boxes[:, 1]) - bias, 0)
            x2 = np.minimum(max(boxes[:, 2]) + bias, image.shape[1])
            y2 = np.minimum(max(boxes[:, 3]) + bias, image.shape[0])

            local_patch = np.copy(image)[y1 : y2, x1 : x2, ]

            local_patch_enhanced = ESAVA_denoising(local_patch)
            image[y1:y2, x1:x2,] = local_patch_enhanced
        
            return image, boxes
        elif self.type == "LocalPatchESAVA":
            return image, boxes
        
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[-1] == 3 else image
            clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
            image = clahe.apply(image)
            return image, boxes
    
    def _main_process(self, data):
        """
        Process the input data to extract contours and masks for each bounding box.

        Parameters
        ----------
        data : tuple
            Contains the image and a list of bounding boxes to process.

        Returns
        -------
        list_mask : list of np.ndarray
            A list of masks indicating regions of interest for each bounding box.
        contours : list of np.ndarray
            A list of contours for each bounding box.

        Notes
        -----
        - If the type is 'OriginalESAVA', processes the image with the bounding box directly.
        - If the type is 'LocalPatchESAVA', preprocesses the local patch with padding and enhancement.
        - For other types, initializes a boundary and relocates it relative to the bounding box.
        - Uses active contour methods to refine the boundaries and generates mask and contour outputs.
        """

        image, boxes = data
        contours = []
        list_mask = []
        
        for bbox in boxes:
            local_patch_preprocessed = self.enhancer(
                data = {
                    "image": image,
                    "boxes": bbox
                }
            ) if self.enhancer is not None else image
            
            if self.type == "OriginalESAVA":
                data = {
                    "image": image,
                    "bounding_box": bbox
                }
            elif self.type == "LocalPatchESAVA":
                padding = 10
                x1, y1, x2, y2 = bbox 
                padded_x1 = x1 - padding
                padded_y1 = y1 - padding
                padded_x2 = x2 + padding
                padded_y2 = y2 + padding
                
                data = {
                    "image": image,
                    "enhanced_local_patch": local_patch_preprocessed,
                    "bounding_box": bbox,
                    "paddded_bounding_box": [padded_x1, padded_y1, padded_x2, padded_y2] 
                }
            else:
                inittial_boundary = np.array(
                    initialize_rotated_ellipse_boundary(bbox, padding = self.config.get("padding"), angle = self.config.get("angle")
                    )
                ) # todo: cho nay can check lai xem co dynamic angle dc ko
                re_location_inittital_boundary = inittial_boundary - np.array(
                    [bbox[1], bbox[0]]
                )
                
                data = {
                    "image": local_patch_preprocessed,
                    "snake": re_location_inittital_boundary,
                }
            
            activated_contour_output = self.active_contour(data)
            
            if activated_contour_output.shape[-1 ]== 2:
                decoded_activated_contour = activated_contour_output + np.array([bbox[1], bbox[0]])
                
                contours.append(decoded_activated_contour)
                
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                contour = decoded_activated_contour.astype(np.int32)
                for idx in range(len(contour)):
                    mask[contour[idx][0], contour[idx][1]] = 1
            else:
                mask = activated_contour_output
                
            
            dilation_mask = morphology.dilation(mask, morphology.square(int(self.config.get("dilation_value"))))
            list_mask.append(dilation_mask)
            
            image = dynamic_inpaint(image, np.uint8(dilation_mask)) if isinstance(self.active_contour, SnakeActiveContour) else image
            
        return list_mask, contours
            
    def _post_process(self, data):
        return data
    
    def handle(self, data: dict):
        list_mask, contours = self.process(data)
        data.update(
            {
                "list_mask": list_mask,
                "contours": contours,
            }
        )
        return super().handle(data)