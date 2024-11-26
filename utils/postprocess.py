#utils/postprocess.py
import cv2
import numpy as np

from utils.info import *
from utils.postprocess_func.output_decoder import (
    ObjDetDecoder,
)
from typing import Dict, List, Sequence, Tuple, Any



# Postprocess for Object Detection
class ObjDetPostprocess:
    def __init__(
        self,
    ):
        self.postprocess_func = ObjDetDecoder("yolov8n")
        self.class_names = CLASS_NAMES

    def __call__(
        self, outputs: List[np.ndarray], contexts: Dict[str, float], img: np.ndarray
    ) -> np.ndarray:
        ## Consider batch 1

        predictions = self.postprocess_func(outputs, contexts, img.shape[:2])
        assert len(predictions) == 1, f"{len(predictions)}!=1"

        predictions = predictions[0]
        num_prediction = predictions.shape[0]

        if num_prediction == 0:
            return img

        bboxed_img = draw_bbox(img, predictions, self.class_names)
        return bboxed_img


# Draw Output on an Original Image
def draw_bbox(
    img: np.ndarray, predictions: np.ndarray, class_names: List[str]
) -> np.ndarray:
    ## Draw box on org image
    for prediction in predictions:
        mbox = [int(i) for i in prediction[:4]]
        score = prediction[4]
        class_id = int(prediction[5])

        color = COLORS[class_id % len(COLORS)]
        label = f"{class_names[class_id]} {int(score*100)}%"
        if len(prediction) != 6:
            tracking_id = int(prediction[-1]) % len(COLORS)
            color = COLORS[tracking_id]
            label = f"{class_names[class_id]}_{tracking_id} {int(score*100)}%"
        img = plot_one_box(mbox, img, color, label)

    return img


def plot_one_box(
    box: List[int],
    img: np.ndarray,
    color: Tuple[int, int, int],
    label: str,
    line_thickness: int = None,
) -> np.ndarray:
    tl = line_thickness or round(0.003 * max(img.shape[0:2])) + 1
    tf = max(tl - 1, 1)
    c1, c2 = (box[0], box[1]), (box[2], box[3])
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1)  # filled
    cv2.putText(
        img,
        label,
        (c1[0], c1[1] - 2),
        0,
        tl / 3,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return img