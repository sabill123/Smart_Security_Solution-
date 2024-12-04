import cv2
import numpy as np
from typing import Any, Dict, List, Tuple

class YOLOPreProcessor:
    def __init__(self, input_size=(640, 640)):
        self.input_size = input_size

    def __call__(self, image: np.ndarray, new_shape: Tuple[int, int] = None, tensor_type: str = "uint8"):
        if new_shape is None:
            new_shape = self.input_size

        # 원본 이미지 크기 유지하면서 패딩
        img, preproc_params = letterbox(image, new_shape, auto=False, scaleFill=False)
        
        # 채널 순서 변경
        img = img.transpose([2, 0, 1])[::-1]
        
        contexts = {
            "ratio": preproc_params[0],
            "pad": preproc_params[1],
            "original_shape": image.shape[:2]
        }

        # uint8 타입으로 통일
        input_data = np.ascontiguousarray(np.expand_dims(img, 0), dtype=np.uint8)
        
        return input_data, contexts

def letterbox(image: np.ndarray, new_shape: Tuple[int,int], color=(114,114,114), auto=True, scaleFill=False):
    h, w = image.shape[:2]
    if h == new_shape[0] and w == new_shape[1]:
        return image, (1.0, (0, 0))
    
    # 비율 계산
    ratio = min(new_shape[0]/h, new_shape[1]/w)
    if not auto:  # 스케일업 방지
        ratio = min(ratio, 1.0)

    # 새로운 크기 계산
    new_unpad = int(round(ratio * w)), int(round(ratio * h))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if not scaleFill:  # 패딩 균등 분배
        dw, dh = dw/2, dh/2

    # 리사이즈
    if ratio != 1.0:
        interpolation = cv2.INTER_LINEAR if ratio > 1 else cv2.INTER_AREA
        image = cv2.resize(image, new_unpad, interpolation=interpolation)

    # 패딩 적용
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return image, (ratio, (dw, dh))