#utils/postprocess_func/nms.py

from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

import torch.nn.functional as F
from utils.postprocess_func.cbox_decode import yolov5_box_decode, yolov8_box_decode
from utils.postprocess_func.nms import non_max_suppression


## Base YOLO Decoder
class YOLO_Decoder:
    """
    Base class for yolo output decoder, providing foundational attributes.

    Attributes:
        conf_thres (float): confidence score threshold for outputs.
        iou_thres (float): iou score threshold for outputs.
        tracker (ByteTrack | None) : tracker object when use traking algorithm.
        anchors (list | none) : anchor values (anchor for yolov8 or yolov9 is None).
        stride (list) : stride values for yolo.
    """

    def __init__(
        self,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        anchors: Union[List[List[int]], None] = None,
    ):
        self.iou_thres = float(iou_thres)
        self.conf_thres = float(conf_thres)
        self.anchors, self.stride = (
            get_anchors(anchors)
            if anchors[0] is not None
            else (None, np.array([(2 ** (i + 3)) for i in range(3)], dtype=np.float32))
        )


## YOLO Object Detection Decoder
class ObjDetDecoder(YOLO_Decoder):
    """
    A integrated version of the object detection decoder class for YOLO, adding nms operator and scaling operators.

    Attributes:
        model_name (str)  : a model name of yolo (e.g., yolov8s, yolov8m ...)
        conf_thres (float): confidence score threshold for outputs.
        iou_thres (float): iou score threshold for outputs.
        anchors (list | none) : anchor values (anchor for yolov8 or yolov9 is None).
        use_trakcer (bool): whether to use tracking algorithm.

    Methods:
        __call__(output, context, org_input_shape): Decode the result of YOLO Model (Object Detection)

    Usage:
        decoder = ObjDetDecoder(model_name, conf_thres, iou_thres, anchors)
        output = decoder(model_outputs, contexts, org_input_shape)
    """

    def __init__(
        self,
        model_name: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        anchors: Union[List[List[int]], None] = [None],
    ):
        super().__init__(conf_thres, iou_thres, anchors)
        self.box_decoder = (
            BoxDecoderYOLOv8(self.stride, self.conf_thres)
            if check_model(model_name)
            else BoxDecoderYOLOv5(self.stride, self.conf_thres, self.anchors)
        )
        self.model_name = model_name

    def __call__(
        self,
        model_outputs: List[np.ndarray],
        contexts,
        org_input_shape: Tuple[int, int],
    ):
        boxes_dec = self.box_decoder(model_outputs)
        outputs = non_max_suppression(boxes_dec, self.iou_thres)
        predictions = []

        ratio, dwdh = contexts["ratio"], contexts["pad"]
        for _, prediction in enumerate(outputs):
            prediction[:, :4] = scale_coords(
                prediction[:, :4], ratio, dwdh, org_input_shape
            )  # Box Result
            predictions.append(prediction)
        return predictions

class CDecoderBase:
    """
    Base class for decoder, providing foundational attributes.

    Attributes:
        conf_thres (float): confidence score threshold for outputs.
        anchors (list | none) : anchor values (anchor for yolov8 or yolov9 is None).
        stride (list) : stride values for yolo.
    """

    def __init__(self, stride: np.ndarray, conf_thres: float, anchors=None) -> None:
        self.stride = stride
        self.conf_thres = conf_thres
        self.anchors = anchors
        self.reg_max = 16

class BoxDecoderYOLOv8(CDecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, feats: List[np.ndarray], step: int = 2):
        feats_box, feats_cls = feats[0::step], feats[1::step]
        feats_extra = None
        if step == 3:
            feats_extra = feats[2::step]

        out_boxes_batched = yolov8_box_decode(
            self.stride,
            self.conf_thres,
            self.reg_max,
            feats_box,
            feats_cls,
            feats_extra,
        )
        return out_boxes_batched


class BoxDecoderYOLOv5(CDecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, feats: List[np.ndarray]):
        out_boxes_batched = yolov5_box_decode(
            self.anchors, self.stride, self.conf_thres, feats
        )
        return out_boxes_batched


## Useful functions for output decoder


def check_model(model_name: str) -> bool:
    if "yolov8" in model_name or "yolov9" in model_name or "yolov10" in model_name:
        return True
    return False


def scale_coords(
    coords: List[np.ndarray],
    ratio: float,
    pad: Tuple[float, float],
    org_input_shape: Tuple[int, int],
    step: int = 2,
) -> np.ndarray:
    ## Scale the result values to fit original image
    coords[:, 0::step] = (1 / ratio) * (coords[:, 0::step] - pad[0])
    coords[:, 1::step] = (1 / ratio) * (coords[:, 1::step] - pad[1])

    ## Clip out-of-bounds values
    coords[:, 0::step] = np.clip(coords[:, 0::step], 0, org_input_shape[1])
    coords[:, 1::step] = np.clip(coords[:, 1::step], 0, org_input_shape[0])

    return coords


def get_anchors(anchors):
    num_layers = len(anchors)
    anchors = np.reshape(np.array(anchors, dtype=np.float32), (num_layers, -1, 2))
    stride = np.array([2 ** (i + 3) for i in range(num_layers)], dtype=np.float32)
    anchors /= np.reshape(stride, (-1, 1, 1))
    return anchors, stride
#################################################