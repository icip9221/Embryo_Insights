import cv2
import numpy as np
import onnxruntime

from abstract import AbstractOnnxInference

from .utils import draw_detections, multiclass_nms, xywh2xyxy


class YOLOv8DETONNX(AbstractOnnxInference):
    """
    A YOLOv8 object detector using ONNX Runtime for inference.

    This class wraps a YOLOv8 ONNX model to perform object detection
    on input images using configurable confidence and IoU thresholds.
    It handles preprocessing, inference, postprocessing (including NMS),
    and optional visualization of detections.

    Parameters:
    - path (str): Path to the YOLOv8 ONNX model file.
    - conf_thres (float): Confidence threshold to filter low-score predictions. Default is 0.7.
    - iou_thres (float): IoU threshold for Non-Maximum Suppression (NMS). Default is 0.5.

    Methods:
    ----------
    __call__(image):
        Runs object detection on the input image and returns boxes, scores, and class_ids.

    initialize_model(path):
        Loads the ONNX model and initializes session with available execution providers.

    prepare_input(image):
        Preprocesses the input image:
        - Converts BGR to RGB.
        - Resizes to model input size.
        - Normalizes pixel values to [0, 1].
        - Converts to NCHW format and float32 tensor.

    detect_objects(image):
        Applies the full detection pipeline (preprocess → inference → postprocess).
        Returns final filtered boxes, scores, and class IDs.

    inference(input_tensor):
        Calls the parent inference method with the prepared input tensor.

    process_output(output):
        Postprocesses raw model output:
        - Applies confidence filtering.
        - Extracts class IDs.
        - Converts coordinates.
        - Applies multiclass NMS.
        Returns filtered results.

    extract_boxes(predictions):
        Extracts bounding box coordinates from YOLOv8 output (in `xywh`) and
        converts them to `xyxy` format after rescaling.

    rescale_boxes(boxes):
        Rescales bounding boxes from model input size to the original image size.

    draw_detections(image, draw_scores=True, mask_alpha=0.4):
        Draws final detections on the image using bounding boxes and labels.
        Returns the image with overlaid detections.

    get_input_details():
        Returns input tensor metadata from ONNX model.

    get_output_details():
        Returns output tensor metadata from ONNX model.

    Returns:
    - boxes (np.ndarray): Detected bounding boxes in xyxy format.
    - scores (np.ndarray): Confidence scores of detected objects.
    - class_ids (np.ndarray): Predicted class indices.
    
    Example:
    --------
    >>> detector = YOLOv8DETONNX("yolov8.onnx", conf_thres=0.5)
    >>> boxes, scores, class_ids = detector(image)
    >>> result = detector.draw_detections(image)
    """
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        super(YOLOv8DETONNX, self).__init__()
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path, providers=onnxruntime.get_available_providers())
        # Get model info
        self.get_input_details()
        self.get_output_details()


    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        return super().inference(input_tensor)

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        return super().get_input_details()

    def get_output_details(self):
        return super().get_output_details()
