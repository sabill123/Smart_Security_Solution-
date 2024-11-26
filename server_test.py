#잘되는버전.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import Optional, Dict, AsyncIterator, AsyncGenerator
import threading
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import asyncio
import numpy as np
import time
import logging
from furiosa.runtime import create_runner
from utils.preprocess import YOLOPreProcessor
from utils.postprocess_func.output_decoder import ObjDetDecoder
import pickle
import struct
from collections import defaultdict, deque
from enum import Enum
import gc




class MosaicMode(Enum):
    ALL = "all"              # 모든 사람 모자이크
    NONE = "none"           # 탐지된 사람 모자이크 제외
    SELECTIVE = "selective"  # 선택적 모자이크 제외
    AUTO = "auto"           # 자동 주체 판단 모드

class VideoProcessingServer:
    def __init__(self, model_path: str, host: str = '0.0.0.0', port: int = 8485, reference_face: str = None):
        self.host = host
        self.port = port
        self.model_path = model_path
        # 640x640 고정 크기로 YOLOPreProcessor 초기화
        self.preprocessor = YOLOPreProcessor(input_size=(640, 640))
        # threshold 값 조정
        self.decoder = ObjDetDecoder("yolov8n", conf_thres=0.05, iou_thres=0.65)  # 더 낮은 confidence, 더 높은 IOU threshold
        self.running = True
        self.total_detections = 0
        self.start_time = None
        self.tracks = {}
        self.next_id = 1
        
         # 프레임 처리 관련 설정
        self.frame_skip = 0
        self.process_every_n_frames = 2
        self.last_process_time = time.time()
        self.min_process_interval = 1.0 / 30.0  # 최소 30fps 보장
        
        # 메모리 관리 설정
        self.max_active_tracks = 30
        self.max_history_size = 100
        self.cleanup_interval = 50  # 프레임
        
        # 성능 메트릭 초기화를 생성자에서 수행
        self.performance_metrics = {
            'start_time': time.time(),
            'frame_count': 0,
            'total_process_time': 0.0,
            'last_fps_update': time.time(),
            'process_times': deque(maxlen=30)
        }
        
        # 모자이크 관련 설정
        self.mosaic_mode = MosaicMode.ALL
        self.excluded_ids = set()  # 모자이크 제외할 ID 목록
        self.auto_mode_threshold = 0.6  # 자동 모드에서 사용할 임계값
        self.selective_mode_subjects = set()  # selective 모드에서 선택된 주체들
        
        self.session_data = {}  # 세션 데이터 저장을 위한 딕셔너리 추가

        
        # 주체 처리 관련 변수들
        self.first_frame_processed = False
        self.position_history = defaultdict(list)
        self.frame_presence = defaultdict(int)
        self.subject_features = {}  # 주체 얼굴 특징
        self.subject_history = {}   # 주체들의 위치 이력
        self.subject_confidence = defaultdict(float)  # 각 ID별 주체 신뢰도
        
        self.face_features_db = {}  # ID별 얼굴 특징 저장
        self.face_history = {}      # ID별 얼굴 이력 관리
        self.feature_similarity_threshold = 0.7  # 얼굴 유사도 임계값
        self.max_features_per_id = 5  # ID당 최대 저장할 특징 수
        
        # 파라미터 설정
        self.similarity_threshold = 0.6  # 얼굴 유사도 임계값
        self.position_buffer = 10       # 위치 이력 버퍼 크기
        self.MIN_FRAMES = 30           # 최소 등장 프레임 수
        self.MIN_SIZE_RATIO = 0.25     # 최소 크기 비율
        self.MAX_MOVEMENT = 0.4        # 최대 이동 거리
        
        # 웹 스트리밍용 속성 추가
        self.latest_frame = None
        self.web_clients: Dict[str, bool] = {}
        self.web_app = FastAPI()
        self.templates = Jinja2Templates(directory="templates")
        self.setup_web_server()
        
        
        self.tracks = {}
        self.next_id = 1
        self.mosaic_mode = MosaicMode.ALL
        self.excluded_ids = set()
        
        # 자원 관리를 위한 락
        self._cleanup_lock = asyncio.Lock()
        
        if reference_face and os.path.exists(reference_face):
            self.add_reference_face(reference_face, "main_subject")
        
        print(f"서버 초기화 완료 - 모델: {model_path}")

    def extract_face_feature(self, face_roi):
        """안전한 얼굴 특징 추출"""
        try:
            if face_roi is None or face_roi.size == 0:
                return None
                
            # 안전한 크기 조정
            if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                return None
                
            # 깊은 복사로 원본 데이터 보호
            face = face_roi.copy()
            
            # 크기 조정 전 타입 및 값 검사
            face = np.ascontiguousarray(face)
            if not face.flags['C_CONTIGUOUS']:
                face = np.ascontiguousarray(face)
                
            face = cv2.resize(face, (64, 64))
            if len(face.shape) == 3:
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face
                
            feature = face_gray.flatten().astype(np.float32)
            norm = np.linalg.norm(feature)
            if norm == 0:
                return None
            return feature / norm
            
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None

    def update_face_db(self, track_id: int, face_roi: np.ndarray):
        """안전한 얼굴 DB 업데이트"""
        try:
            if face_roi is None or face_roi.size == 0:
                return
                
            # 얼굴 특징 추출
            face_feature = self.extract_face_feature(face_roi)
            if face_feature is None:
                return
                
            # DB 크기 제한
            max_stored_features = 1000  # 전체 저장 가능한 최대 특징 수
            current_total_features = sum(len(features) for features in self.face_features_db.values())
            
            if current_total_features >= max_stored_features:
                # 가장 오래된 특징 제거
                oldest_time = float('inf')
                oldest_id = None
                for tid, time_val in self.face_history.items():
                    if time_val < oldest_time:
                        oldest_time = time_val
                        oldest_id = tid
                if oldest_id and oldest_id in self.face_features_db:
                    del self.face_features_db[oldest_id]
                    del self.face_history[oldest_id]
            
            # 새로운 특징 추가
            if track_id not in self.face_features_db:
                self.face_features_db[track_id] = []
                
            self.face_features_db[track_id].append(face_feature)
            if len(self.face_features_db[track_id]) > self.max_features_per_id:
                self.face_features_db[track_id].pop(0)
                
            self.face_history[track_id] = time.time()
            
        except Exception as e:
            print(f"Face DB update error: {str(e)}")

    def find_matching_id(self, face_roi):
        """안전한 ID 매칭"""
        try:
            if face_roi is None or face_roi.size == 0:
                return None
                
            current_feature = self.extract_face_feature(face_roi)
            if current_feature is None:
                return None
                
            best_match_id = None
            best_similarity = self.feature_similarity_threshold
            current_time = time.time()
            
            # 임시 목록 생성으로 dictionary 변경 중 오류 방지
            track_items = list(self.face_features_db.items())
            
            for track_id, features in track_items:
                if current_time - self.face_history.get(track_id, 0) > 60:
                    continue
                    
                if not features:
                    continue
                    
                # 안전한 유사도 계산
                try:
                    similarities = [self.calculate_similarity(current_feature, feat) for feat in features]
                    max_similarity = max(similarities) if similarities else 0.0
                    
                    if max_similarity > best_similarity:
                        best_similarity = max_similarity
                        best_match_id = track_id
                except Exception as e:
                    print(f"Similarity computation error for track {track_id}: {str(e)}")
                    continue
            
            return best_match_id
                
        except Exception as e:
            print(f"ID matching error: {str(e)}")
            return None
        
    async def async_find_matching_id(self, face_roi: np.ndarray) -> Optional[int]:
        """ID 매칭 비동기 버전"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.find_matching_id, face_roi
            )
        except Exception as e:
            logging.error(f"Async ID matching error: {str(e)}")
            return None
    
    def update_mosaic_settings(self, mode: str, selected_ids: list = None):
        """모자이크 설정 업데이트"""
        try:
            mode = mode.lower()
            self.mosaic_mode = MosaicMode(mode)
            logging.info(f"Mosaic mode updated to: {mode}")

            if mode == "all":
                self.excluded_ids.clear()
            elif mode == "none":
                self.excluded_ids = set(self.tracks.keys())
            elif mode == "selective" and selected_ids is not None:
                self.excluded_ids = set(selected_ids)
            elif mode == "auto":
                # auto 모드에서는 excluded_ids를 자동으로 업데이트
                self.excluded_ids = self._identify_main_subjects()
            
            logging.info(f"Excluded IDs updated: {self.excluded_ids}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to update mosaic settings: {e}")
            return False
        
    def _identify_main_subjects(self) -> set:
        """AUTO 모드에서 주요 피사체 식별"""
        main_subjects = set()
        if not self.tracks:
            return main_subjects

        for track_id, track_info in self.tracks.items():
            # 1. 중앙 위치 점수
            center_score = self._calculate_center_score(track_info['bbox'])
            
            # 2. 크기 점수
            size_score = self._calculate_size_score(track_info['bbox'])
            
            # 3. 지속성 점수
            duration_score = self._calculate_duration_score(track_id)
            
            # 종합 점수 계산
            total_score = (center_score * 0.4 + 
                         size_score * 0.3 + 
                         duration_score * 0.3)
            
            if total_score > self.auto_mode_threshold:
                main_subjects.add(track_id)
                
        return main_subjects

    def should_apply_mosaic(self, track_id: int) -> bool:
        """모자이크 적용 여부 결정"""
        try:
            if self.mosaic_mode == MosaicMode.ALL:
                return True
            elif self.mosaic_mode == MosaicMode.NONE:
                return False
            elif self.mosaic_mode == MosaicMode.SELECTIVE:
                return track_id not in self.excluded_ids
            elif self.mosaic_mode == MosaicMode.AUTO:
                return track_id not in self.excluded_ids
            return True
        except Exception as e:
            logging.error(f"Error in should_apply_mosaic: {e}")
            return True

    def get_host_status(self, track_id: int) -> str:
        """각 모드별 host 상태 결정"""
        try:
            # ALL 모드에서는 무조건 non-host
            if self.mosaic_mode == MosaicMode.ALL:
                return 'n'
            elif self.mosaic_mode == MosaicMode.NONE:
                return 'y'  # 모든 객체를 host로 표시
            elif self.mosaic_mode == MosaicMode.SELECTIVE:
                return 'y' if track_id in self.excluded_ids else 'n'
            elif self.mosaic_mode == MosaicMode.AUTO:
                return 'y' if track_id in self.excluded_ids else 'n'
            return 'n'
        except Exception as e:
            logging.error(f"Host status error: {str(e)}")
            return 'n'

    def add_reference_face(self, face_image_path: str, label: str = "main_subject"):
        """사전 등록 얼굴 이미지 추가"""
        if os.path.exists(face_image_path):
            face = cv2.imread(face_image_path)
            if face is not None:
                face_gray = cv2.cvtColor(cv2.resize(face, (64, 64)), cv2.COLOR_BGR2GRAY)
                self.subject_features[label] = face_gray
                self.subject_history[label] = deque(maxlen=self.position_buffer)
                print(f"주체 등록 완료: {label}")
                return True
        return False

    def calculate_similarity(self, feature1, feature2):
        """안전한 유사도 계산"""
        try:
            if feature1 is None or feature2 is None:
                return 0.0
                
            if feature1.size != feature2.size:
                return 0.0
                
            similarity = np.dot(feature1, feature2)
            return max(0.0, min(1.0, similarity))  # 0과 1 사이로 제한
            
        except Exception as e:
            print(f"Similarity calculation error: {str(e)}")
            return 0.0

        
    def calculate_iou(self, box1, box2):
        """단일 박스 쌍의 IOU 계산"""
        # 박스 좌표 추출
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교차 영역 계산
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # 교차 영역이 없는 경우
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        # 교차 영역 넓이 계산
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 각 박스의 넓이 계산
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # IOU 계산
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0.0
        
        return iou

    def calculate_iou_matrix(self, boxes1, boxes2):
        """벡터화된 IOU 계산"""
        # boxes1: (N,4), boxes2: (M,4) -> output: (N,M)
        x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
        
        # 교차 영역 계산
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        
        intersection = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        
        # 각 박스의 면적 계산
        box1_area = (x12 - x11) * (y12 - y11)
        box2_area = (x22 - x21) * (y22 - y21)
        
        union = box1_area + np.transpose(box2_area) - intersection
        
        return intersection / (union + 1e-6)

    def greedy_matching(self, iou_matrix, thresh):
        """Greedy matching for tracking"""
        matched_indices = []
        
        if iou_matrix.size == 0:
            return matched_indices
            
        for _ in range(min(iou_matrix.shape)):
            # Find highest IOU
            max_iou = np.max(iou_matrix)
            if max_iou < thresh:
                break
                
            # Get indices of max IOU
            det_idx, track_idx = np.unravel_index(
                np.argmax(iou_matrix), iou_matrix.shape
            )
            
            matched_indices.append([det_idx, track_idx])
            
            # Remove matched detection and track
            iou_matrix[det_idx, :] = 0
            iou_matrix[:, track_idx] = 0
            
        return matched_indices

    def update_position_history(self, track_id, bbox, frame_shape):
        """위치 이력 업데이트"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2 / frame_shape[1]
        center_y = (y1 + y2) / 2 / frame_shape[0]
        size_ratio = (x2 - x1) * (y2 - y1) / (frame_shape[0] * frame_shape[1])
        
        if track_id not in self.position_history:
            self.position_history[track_id] = deque(maxlen=self.position_buffer)
        self.position_history[track_id].append((center_x, center_y, size_ratio))
        self.frame_presence[track_id] = self.frame_presence.get(track_id, 0) + 1

    async def update_tracks(self, detections, frame_number: int, frame: np.ndarray):
        """메모리 안전성이 개선된 트래킹 로직 - 비동기 버전"""
        try:
            # 기존 트랙 상태 체크 - 트래킹 유지 시간 증가
            active_tracks = {
                track_id: track 
                for track_id, track in self.tracks.items()
                if frame_number - track['last_seen'] <= 60  # 더 긴 트래킹 유지
            }

            # 매칭 처리
            matched_track_ids = set()
            matched_det_indices = set()

            # 모든 detection에 대해 처리
            for det_idx, detection in enumerate(detections):
                best_iou = 0.2  # 더 낮은 IOU 임계값
                best_track_id = None

                # 활성화된 트랙과 IOU 계산
                for track_id, track in active_tracks.items():
                    if track_id in matched_track_ids:
                        continue

                    iou = self.calculate_iou(track['bbox'], detection[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id

                # 매칭된 트랙 업데이트
                if best_track_id is not None:
                    self.tracks[best_track_id].update({
                        'bbox': detection[:4],
                        'confidence': max(detection[4], self.tracks[best_track_id].get('confidence', 0)),  # 높은 confidence 유지
                        'last_seen': frame_number
                    })
                    matched_track_ids.add(best_track_id)
                    matched_det_indices.add(det_idx)

            # 새로운 검출에 대한 배치 처리
            new_detections = [(det_idx, detection) for det_idx, detection in enumerate(detections) 
                            if det_idx not in matched_det_indices]
            
            if new_detections:
                # 얼굴 ROI 배치 추출
                face_rois = []
                for det_idx, detection in new_detections:
                    x1, y1, x2, y2 = map(int, detection[:4])
                    face_roi = self.extract_face_roi(frame, x1, y1, x2, y2)
                    face_rois.append((det_idx, face_roi))

                # 유효한 얼굴 ROI에 대해서만 ID 매칭 시도
                tasks = []
                for det_idx, face_roi in face_rois:
                    if face_roi is not None:
                        task = asyncio.create_task(self.async_find_matching_id(face_roi))
                        tasks.append((det_idx, task))

                # 비동기 작업 결과 처리
                if tasks:
                    completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                    
                    for (det_idx, _), matching_id in zip(tasks, completed_tasks):
                        try:
                            if isinstance(matching_id, Exception):
                                continue
                                
                            detection = detections[det_idx]
                            track_id = matching_id if matching_id is not None else self.next_id
                            if matching_id is None:
                                self.next_id += 1

                            self.tracks[track_id] = {
                                'bbox': detection[:4],
                                'confidence': detection[4],
                                'last_seen': frame_number,
                                'first_seen': frame_number
                            }

                            # 얼굴 특징 DB 업데이트
                            x1, y1, x2, y2 = map(int, detection[:4])
                            face_roi = self.extract_face_roi(frame, x1, y1, x2, y2)
                            if face_roi is not None:
                                await asyncio.get_event_loop().run_in_executor(
                                    None, self.update_face_db, track_id, face_roi
                                )

                        except Exception as e:
                            logging.error(f"Error processing detection {det_idx}: {str(e)}")

        except Exception as e:
            logging.error(f"Track update error: {str(e)}")


    def extract_face_roi(self, frame, x1, y1, x2, y2):
        """개선된 얼굴 영역 추출"""
        try:
            person_height = y2 - y1
            if person_height <= 0:
                return None
                
            head_height = int(person_height * 0.25)
            head_width = int((x2 - x1) * 0.6)
            head_x1 = x1 + ((x2 - x1) - head_width) // 2
            head_x2 = head_x1 + head_width
            head_y2 = min(y1 + head_height, frame.shape[0])
            head_y1 = max(0, y1)
            
            if head_x2 <= head_x1 or head_y2 <= head_y1:
                return None

            head_x1 = max(0, head_x1)
            head_x2 = min(frame.shape[1], head_x2)
            
            if head_x2 > head_x1 and head_y2 > head_y1:
                return frame[head_y1:head_y2, head_x1:head_x2].copy()
            return None
        except:
            return None

    def apply_mosaic(self, result, x1, y1, x2, y2):
        try:
            # 모자이크 영역 안정화를 위한 시간적 스무딩 추가
            if not hasattr(self, 'mosaic_history'):
                self.mosaic_history = {}
                
            track_id = self.get_track_id_for_bbox([x1, y1, x2, y2])
            if track_id not in self.mosaic_history:
                self.mosaic_history[track_id] = {
                    'boxes': [],
                    'frames_stable': 0
                }
                
            history = self.mosaic_history[track_id]
            history['boxes'].append([x1, y1, x2, y2])
            if len(history['boxes']) > 5:  # 최근 5프레임만 유지
                history['boxes'].pop(0)
                
            # 박스 위치 안정성 체크
            stable = True
            if len(history['boxes']) >= 3:
                for i in range(1, len(history['boxes'])):
                    prev_box = history['boxes'][i-1]
                    curr_box = history['boxes'][i]
                    if any(abs(p - c) > 20 for p, c in zip(prev_box, curr_box)):
                        stable = False
                        break
                        
            if stable:
                history['frames_stable'] += 1
            else:
                history['frames_stable'] = 0
                
            # 일정 시간 안정적일 때만 모자이크 적용
            if history['frames_stable'] >= 3:
                # 모자이크 크기 동적 조정
                person_height = y2 - y1
                if person_height < 100:  # 작은 객체
                    mosaic_size = 10
                elif person_height < 200:  # 중간 객체
                    mosaic_size = 15
                else:  # 큰 객체
                    mosaic_size = 20
                
                # 모자이크 영역 최적화
                head_y1 = max(0, y1)
                head_y2 = min(result.shape[0], y1 + int(person_height * 0.3))
                head_width = int((x2 - x1) * 0.7)
                head_x1 = max(0, x1 + ((x2 - x1) - head_width) // 2)
                head_x2 = min(result.shape[1], head_x1 + head_width)

                if head_x2 > head_x1 and head_y2 > head_y1:
                    roi = result[head_y1:head_y2, head_x1:head_x2]
                    h, w = roi.shape[:2]
                    
                    if h > 0 and w > 0:
                        # 모자이크 처리 최적화
                        small = cv2.resize(roi, (max(1, w//mosaic_size), max(1, h//mosaic_size)))
                        blurred = cv2.resize(small, (w, h))
                        
                        # 가우시안 블러 크기 동적 조정
                        blur_size = min(31, max(5, int(person_height * 0.15)))
                        if blur_size % 2 == 0:
                            blur_size += 1
                        blurred = cv2.GaussianBlur(blurred, (blur_size, blur_size), 0)
                        
                        result[head_y1:head_y2, head_x1:head_x2] = blurred

        except Exception as e:
            print(f"Mosaic application error: {str(e)}")
            
    def calculate_size_score(self, bbox, frame_shape):
        """객체 크기 점수 계산"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        frame_area = frame_shape[0] * frame_shape[1]
        
        # 크기 비율 계산
        size_ratio = area / frame_area
        
        # 적정 크기 범위 내에서 점수 계산
        if 0.05 < size_ratio < 0.3:
            return 1.0
        elif size_ratio <= 0.05:
            return size_ratio / 0.05
        else:
            return max(0, 1 - (size_ratio - 0.3) / 0.3)

    def calculate_position_score(self, bbox, frame_shape):
        """객체 위치 점수 계산"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2 / frame_shape[1]
        center_y = (y1 + y2) / 2 / frame_shape[0]
        
        # 중앙 영역에 가까울수록 높은 점수
        distance_from_center = np.sqrt(
            (center_x - 0.5) ** 2 + 
            (center_y - 0.5) ** 2
        )
        
        # 정규화된 거리 점수
        position_score = max(0, 1 - distance_from_center)
        
        # 화면 중앙 영역에 있는 경우 보너스 점수
        if 0.3 < center_x < 0.7 and 0.3 < center_y < 0.7:
            position_score *= 1.2
            
        return min(1.0, position_score)

    def calculate_stability_score(self, track_id):
        """객체 안정성 점수 계산"""
        if track_id not in self.position_history:
            return 0.0
            
        history = self.position_history[track_id]
        if len(history) < 3:
            return 0.0
            
        # 이동 거리 변화 계산
        movements = []
        for i in range(1, len(history)):
            prev_x, prev_y, _ = history[i-1]
            curr_x, curr_y, _ = history[i]
            movement = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            movements.append(movement)
        
        # 이동 안정성 점수
        avg_movement = np.mean(movements)
        movement_score = max(0, 1 - avg_movement / 0.1)  # 0.1은 임계값
        
        # 크기 변화 안정성
        sizes = [s for _, _, s in history]
        size_std = np.std(sizes)
        size_score = max(0, 1 - size_std / 0.1)  # 0.1은 임계값
        
        # 프레임 지속성 점수
        duration_score = min(len(history) / self.MIN_FRAMES, 1.0)
        
        # 최종 안정성 점수 계산
        stability_score = (
            movement_score * 0.4 +
            size_score * 0.3 +
            duration_score * 0.3
        )
        
        return stability_score

    def integrate_subject_detection(self, track_id: int, bbox: tuple, face_roi: np.ndarray, frame_shape: tuple, frame_number: int) -> float:
        """강화된 주체 판단 시스템"""
        score = 0.0
        x1, y1, x2, y2 = bbox

        # 1. 주체 특성 기준 강화
        size_score = self.calculate_size_score(bbox, frame_shape)
        position_score = self.calculate_position_score(bbox, frame_shape)
        stability_score = self.calculate_stability_score(track_id)
        
        # 크기가 큰 객체에 가중치 부여
        relative_size = (x2 - x1) * (y2 - y1) / (frame_shape[0] * frame_shape[1])
        if relative_size > 0.1:  # 화면의 10% 이상 차지
            size_score *= 1.5
        
        # 화면 중앙에 있는 객체에 가중치 부여
        center_x = (x1 + x2) / 2 / frame_shape[1]
        if 0.3 < center_x < 0.7:  # 화면 중앙 영역
            position_score *= 1.3
        
        # 최종 점수 계산
        final_score = (
            size_score * 0.3 +
            position_score * 0.3 +
            stability_score * 0.4
        )
        
        # 연속성 검사를 위한 임계값 조정
        if track_id in self.excluded_ids:
            threshold = 0.4  # 주체로 인식된 경우 더 관대하게
        else:
            threshold = 0.6  # 새로운 주체 판단은 더 엄격하게
        
        # 연속 프레임 카운터 업데이트    
        if not hasattr(self, 'subject_frame_counter'):
            self.subject_frame_counter = defaultdict(int)
        
        if final_score > threshold:
            self.subject_frame_counter[track_id] += 1
        else:
            self.subject_frame_counter[track_id] = max(0, self.subject_frame_counter[track_id] - 1)
        
        # 주체 판단 로직 개선
        is_subject = self.subject_frame_counter[track_id] >= 5
        if is_subject and track_id not in self.excluded_ids:
            self.excluded_ids.add(track_id)
            print(f"New subject detected - ID: {track_id}, Score: {final_score:.2f}")
        elif not is_subject and track_id in self.excluded_ids:
            if self.subject_frame_counter[track_id] < 3:
                self.excluded_ids.discard(track_id)
                print(f"Subject lost - ID: {track_id}")
        
        return final_score

    def has_subject_characteristics(self, bbox: tuple, frame_shape: tuple) -> bool:
        """주체가 될 수 있는 특성 체크"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        frame_area = frame_shape[0] * frame_shape[1]
        
        size_ratio = area / frame_area
        if not (0.05 < size_ratio < 0.3):
            return False
            
        center_x = (x1 + x2) / 2 / frame_shape[1]
        if not (0.2 < center_x < 0.8):
            return False
            
        return True
    
    def get_track_id_for_bbox(self, bbox):
        """바운딩 박스에 해당하는 track_id 찾기"""
        for track_id, track_info in self.tracks.items():
            if self.calculate_iou(track_info['bbox'], bbox) > 0.5:
                return track_id
        return None

    def predict_next_bbox(self, history):
        """트래킹 히스토리 기반 다음 위치 예측"""
        if len(history) < 2:
            return history[-1]['bbox']
        
        last_box = history[-1]['bbox']
        prev_box = history[-2]['bbox']
        
        # 이동 벡터 계산
        dx = (last_box[2] + last_box[0])/2 - (prev_box[2] + prev_box[0])/2
        dy = (last_box[3] + last_box[1])/2 - (prev_box[3] + prev_box[1])/2
        
        # 다음 위치 예측
        next_x1 = last_box[0] + dx
        next_y1 = last_box[1] + dy
        next_x2 = last_box[2] + dx
        next_y2 = last_box[3] + dy
        
        return [next_x1, next_y1, next_x2, next_y2]

    def validate_with_hog(self, roi):
        """HOG 기반 사람 검출 검증"""
        try:
            if roi is None or roi.size == 0:
                return 0
                
            # ROI 크기 정규화
            roi = cv2.resize(roi, (64, 128))
            
            # HOG 특징 추출
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # 검출 수행
            found, _ = hog.detectMultiScale(roi, 
                                        winStride=(8,8),
                                        padding=(4,4),
                                        scale=1.05)
            
            return 1.0 if len(found) > 0 else 0.0
        except Exception as e:
            print(f"HOG validation error: {str(e)}")
            return 0.0

    def filter_detections(self, predictions, frame, frame_shape):
        try:
            valid_detections = []
            detections = sorted(predictions[0], key=lambda x: x[4], reverse=True)
            
            # NMS 적용
            boxes = [det[:4] for det in detections]
            scores = [det[4] for det in detections]
            indices = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                0.05,    # 더 낮은 confidence threshold
                0.65     # 더 높은 NMS threshold
            )
            
            for idx in indices:
                if isinstance(idx, (list, tuple)):
                    idx = idx[0]
                    
                det = detections[idx]
                if int(det[5]) != 0:  # person class 확인
                    continue
                    
                confidence = det[4]
                bbox = det[:4]
                x1, y1, x2, y2 = map(int, bbox)
                
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = height / width if width > 0 else 0
                area_ratio = (width * height) / (frame_shape[0] * frame_shape[1])
                
                # 더 완화된 필터링 조건
                if (confidence > 0.05 and                # 매우 낮은 confidence도 허용
                    0.1 < aspect_ratio < 7.0 and        # 더 넓은 종횡비 범위
                    0.00001 < area_ratio < 0.99 and     # 매우 작은 객체도 허용
                    height > 5 and                      # 최소 높이 더 완화
                    width > 4):                         # 최소 너비 더 완화
                    
                    valid_detections.append(det)
                    
            return valid_detections
                
        except Exception as e:
            logging.error(f"Detection filtering error: {e}")
            return []

    def validate_person_detection(self, roi):
        """완화된 사람 검출 검증"""
        try:
            if roi is None or roi.size == 0:
                return False
                
            # 크기가 너무 작은 경우 검증 생략
            if roi.shape[0] < 20 or roi.shape[1] < 15:
                return True
                
            # HSV 색상 검증 (피부색/의류 색상)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 피부색 범위 확장
            lower_skin = np.array([0, 10, 60])
            upper_skin = np.array([25, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # 의류 색상 범위 (넓은 범위)
            cloth_mask = cv2.inRange(hsv, np.array([0, 0, 30]), np.array([180, 255, 255]))
            
            skin_ratio = np.count_nonzero(skin_mask) / skin_mask.size
            cloth_ratio = np.count_nonzero(cloth_mask) / cloth_mask.size
            
            # 완화된 조건
            if (skin_ratio > 0.05 or  # 적은 피부 비율도 허용
                cloth_ratio > 0.3):   # 의류 비율 조건 완화
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Person validation error: {e}")
            return False
        
    async def cleanup_resources(self, frame_number: int):
        """더 적극적인 메모리 정리 - 비동기 버전"""
        try:
            async with self._cleanup_lock:
                current_time = time.time()
                
                # 트랙 개수 제한
                if len(self.tracks) > self.max_active_tracks:
                    # 가장 오래된 트랙 제거
                    sorted_tracks = sorted(
                        self.tracks.items(), 
                        key=lambda x: x[1]['last_seen']
                    )
                    
                    for track_id, _ in sorted_tracks[:-self.max_active_tracks]:
                        await self.cleanup_track_data(track_id)

                # 오래된 데이터 정리
                old_tracks = [
                    track_id for track_id, track in self.tracks.items()
                    if frame_number - track['last_seen'] > 60
                ]
                
                for track_id in old_tracks:
                    await self.cleanup_track_data(track_id)

                # 메모리 최적화
                if frame_number % 50 == 0:
                    gc.collect()
                    
                # 메모리 사용량 모니터링
                if frame_number % 100 == 0:
                    import psutil
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    logging.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

        except Exception as e:
            logging.error(f"Cleanup error: {str(e)}")
                
    async def cleanup_track_data(self, track_id: int):
        """트랙 데이터 정리 - 비동기 버전"""
        try:
            # 트랙 관련 모든 데이터 정리
            if track_id in self.tracks:
                del self.tracks[track_id]
            if track_id in self.position_history:
                del self.position_history[track_id]
            if track_id in self.face_features_db:
                del self.face_features_db[track_id]
            if track_id in self.face_history:
                del self.face_history[track_id]
            if track_id in self.subject_frame_counter:
                del self.subject_frame_counter[track_id]
            if hasattr(self, 'mosaic_buffer') and track_id in self.mosaic_buffer:
                del self.mosaic_buffer[track_id]
                
        except Exception as e:
            logging.error(f"Track cleanup error for ID {track_id}: {str(e)}")
        
    def cleanup_old_data(self, frame_number):
        """더 적극적인 메모리 정리"""
        try:
            current_time = time.time()
            
            # 트랙 개수 제한
            if len(self.tracks) > self.max_active_tracks:
                # 가장 오래된 트랙 제거
                sorted_tracks = sorted(self.tracks.items(), 
                                        key=lambda x: x[1]['last_seen'])
                for track_id, _ in sorted_tracks[:-self.max_active_tracks]:
                    del self.tracks[track_id]
                    
                    # 관련된 모든 데이터 정리
                    if track_id in self.position_history:
                        del self.position_history[track_id]
                    if track_id in self.face_features_db:
                        del self.face_features_db[track_id]
                    if track_id in self.face_history:
                        del self.face_history[track_id]
                    if track_id in self.subject_frame_counter:
                        del self.subject_frame_counter[track_id]
                    if hasattr(self, 'mosaic_buffer') and track_id in self.mosaic_buffer:
                        del self.mosaic_buffer[track_id]

            # 오래된 데이터 정리 (60프레임 이상 지난 데이터)
            for track_id in list(self.tracks.keys()):
                if frame_number - self.tracks[track_id]['last_seen'] > 60:
                    del self.tracks[track_id]
                    
                    if track_id in self.position_history:
                        del self.position_history[track_id]
                    if track_id in self.face_features_db:
                        del self.face_features_db[track_id]
                    if track_id in self.face_history:
                        del self.face_history[track_id]
                    if track_id in self.subject_frame_counter:
                        del self.subject_frame_counter[track_id]
                    if hasattr(self, 'mosaic_buffer') and track_id in self.mosaic_buffer:
                        del self.mosaic_buffer[track_id]

            # 버퍼 크기 제한
            for track_id, history in list(self.position_history.items()):
                if len(history) > self.position_buffer:
                    self.position_history[track_id] = deque(
                        list(history)[-self.position_buffer:],
                        maxlen=self.position_buffer
                    )

            # 얼굴 특징 DB 정리
            if len(self.face_features_db) > self.max_active_tracks * 2:
                # 가장 오래된 특징 데이터 정리
                sorted_features = sorted(
                    self.face_history.items(),
                    key=lambda x: x[1]
                )
                for track_id, _ in sorted_features[:-self.max_active_tracks]:
                    if track_id in self.face_features_db:
                        del self.face_features_db[track_id]
                    if track_id in self.face_history:
                        del self.face_history[track_id]

            # 모자이크 버퍼 정리
            if hasattr(self, 'mosaic_buffer'):
                self.mosaic_buffer = {
                    k: v for k, v in self.mosaic_buffer.items() 
                    if k in self.tracks
                }
            
            # 메모리 최적화
            if frame_number % 50 == 0:  # 50프레임마다
                gc.collect()

        except Exception as e:
            print(f"Cleanup error: {str(e)}")
            
    async def receive_client_settings(self, reader, writer) -> bool:
        """클라이언트 설정 수신 - 개선된 버전"""
        try:
            # 설정 데이터 크기 수신
            size_data = await reader.readexactly(struct.calcsize('Q'))
            settings_size = struct.unpack('Q', size_data)[0]
            
            if settings_size <= 0 or settings_size > 1024*1024:  # 최대 1MB로 제한
                logging.error(f"Invalid settings size received: {settings_size}")
                return False
            
            # 설정 데이터 수신
            settings_data = await reader.readexactly(settings_size)
            try:
                settings = pickle.loads(settings_data)
            except Exception as e:
                logging.error(f"Failed to deserialize settings: {e}")
                return False
            
            # 설정 적용
            mode = settings.get('mosaic_mode', 'auto')
            excluded_ids = settings.get('excluded_ids', [])
            self.update_mosaic_settings(mode, excluded_ids)
            
            # 비디오 크기 정보 수신
            try:
                video_info = await reader.readexactly(struct.calcsize('2i'))
                width, height = struct.unpack('2i', video_info)
                logging.info(f"Received client settings - Video dimensions: {width}x{height}")
                
                # 응답 전송
                response = struct.pack('i', 1)  # 성공 응답
                writer.write(response)
                await writer.drain()
                
            except Exception as e:
                logging.error(f"Failed to receive video dimensions: {e}")
                return False
            
            return True
            
        except asyncio.IncompleteReadError as e:
            logging.error(f"Incomplete read during settings reception: {e}")
            return False
        except Exception as e:
            logging.error(f"Failed to receive client settings: {e}")
            return False

    async def receive_frame(self, reader) -> Optional[np.ndarray]:
        """프레임 수신 및 디코딩"""
        try:
            # 프레임 크기 수신
            size_data = await reader.readexactly(struct.calcsize('Q'))
            frame_size = struct.unpack('Q', size_data)[0]
            
            if frame_size <= 0:
                raise ValueError(f"Invalid frame size: {frame_size}")
            
            # 프레임 데이터 수신
            frame_data = await reader.readexactly(frame_size)
            
            # 프레임 디코딩
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode frame data")
                
            return frame
            
        except asyncio.IncompleteReadError:
            logging.info("Client disconnected during frame reception")
            return None
        except Exception as e:
            logging.error(f"Frame reception error: {str(e)}")
            logging.error(f"Frame size: {frame_size if 'frame_size' in locals() else 'unknown'}")
            logging.error(f"Data received: {len(frame_data) if 'frame_data' in locals() else 'none'}")
            return None

    async def send_frame(self, writer, frame: np.ndarray) -> bool:
        """프레임 전송 with 유효성 검사"""
        try:
            # 프레임 유효성 검사
            if frame is None:
                logging.error("Cannot send None frame")
                return False
                
            if not isinstance(frame, np.ndarray):
                logging.error(f"Invalid frame type: {type(frame)}")
                return False
                
            if frame.size == 0 or len(frame.shape) != 3:
                logging.error(f"Invalid frame shape: {frame.shape}")
                return False
                
            # BGR 이미지 확인 및 변환
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            if len(frame.shape) == 3 and frame.shape[2] == 4:  # BGRA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
            # JPEG 인코딩 전 프레임 복사
            frame_copy = frame.copy()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            
            try:
                _, processed_data = cv2.imencode('.jpg', frame_copy, encode_param)
            except cv2.error as e:
                logging.error(f"OpenCV encoding error: {e}")
                return False
                
            if processed_data is None:
                logging.error("Failed to encode frame")
                return False
                
            processed_bytes = processed_data.tobytes()
            
            # 크기 및 데이터 전송
            writer.write(struct.pack('Q', len(processed_bytes)))
            writer.write(processed_bytes)
            await writer.drain()
            
            return True
            
        except Exception as e:
            logging.error(f"Frame sending error: {str(e)}")
            if isinstance(frame, np.ndarray):
                logging.error(f"Frame info - Shape: {frame.shape}, Type: {frame.dtype}")
            return False

    def cleanup_client(self, writer):
        """클라이언트 연결 정리"""
        try:
            if not writer.is_closing():
                writer.close()
            logging.info("Client connection closed")
        except Exception as e:
            logging.error(f"Error during client cleanup: {str(e)}")

    async def log_session_summary(self, frame_number: int, session_start: float):
        """세션 요약 정보 로깅"""
        try:
            total_time = time.time() - session_start
            avg_fps = frame_number/total_time if total_time > 0 else 0
            
            logging.info("\nSession Summary:")
            logging.info(f"Total time: {total_time:.2f} seconds")
            logging.info(f"Frames processed: {frame_number}")
            logging.info(f"Average FPS: {avg_fps:.2f}")
            
            # 메모리 사용량
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            logging.info(f"Final memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            logging.error(f"Error logging session summary: {str(e)}")

    async def send_processed_frame(self, writer, frame: np.ndarray) -> bool:
        """처리된 프레임 전송"""
        try:
            if frame is None:
                logging.error("Cannot send None frame")
                return False
                
            # 프레임 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            ret, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
            
            if not ret:
                logging.error("Failed to encode processed frame")
                return False
                
            # 인코딩된 데이터 전송
            frame_data = encoded_frame.tobytes()
            writer.write(struct.pack('Q', len(frame_data)))
            writer.write(frame_data)
            await writer.drain()
            
            return True
            
        except Exception as e:
            logging.error(f"Frame sending error: {str(e)}")
            return False
        
    def validate_frame(self, frame: np.ndarray) -> bool:
        """프레임 유효성 검사"""
        if frame is None:
            return False
            
        if not isinstance(frame, np.ndarray):
            return False
            
        if len(frame.shape) != 3:
            return False
            
        if frame.shape[2] != 3:  # BGR 채널 확인
            return False
            
        if frame.dtype != np.uint8:
            return False
            
        return True

    async def cleanup_session(self, writer, session_id: int):
        """세션 정리 및 요약"""
        try:
            # 세션 요약 로깅
            if session_id in self.session_data:
                session = self.session_data[session_id]
                total_time = time.time() - session['start_time']
                frame_count = session['frame_count']
                
                avg_fps = frame_count / total_time if total_time > 0 else 0
                avg_process_time = (sum(session['process_times']) / len(session['process_times']) 
                                  if session['process_times'] else 0)
                
                logging.info("\nSession Summary:")
                logging.info(f"Total time: {total_time:.2f} seconds")
                logging.info(f"Frames processed: {frame_count}")
                logging.info(f"Average FPS: {avg_fps:.2f}")
                logging.info(f"Average processing time: {avg_process_time*1000:.2f}ms")
                
                # 메모리 사용량
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                logging.info(f"Final memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
                
                # 세션 데이터 정리
                del self.session_data[session_id]
            
            # 연결 종료
            if not writer.is_closing():
                writer.close()
            await writer.wait_closed()
            logging.info("Client connection closed")
            
        except Exception as e:
            logging.error(f"Error during session cleanup: {str(e)}")
            # 연결이 아직 열려있다면 강제로 닫기
            try:
                if not writer.is_closing():
                    writer.close()
                    await writer.wait_closed()
            except:
                pass

    async def handle_client(self, reader, writer):
        """클라이언트 처리 메서드"""
        client_address = writer.get_extra_info('peername')
        session_id = id(writer)
        frame_number = 0
        session_start = time.time()
        
        logging.info(f"New client connected from {client_address}")
        
        try:
            # 클라이언트 설정 수신
            if not await self.receive_client_settings(reader, writer):
                logging.error("Failed to receive client settings")
                return
                
            # 메인 처리 루프
            async with create_runner(self.model_path, device="npu0pe0,npu0pe1,npu1pe0,npu1pe1") as runner:
                while True:
                    try:
                        frame = await self.receive_frame(reader)
                        if frame is None:
                            break
                            
                        processed_frame = await self.process_frame(runner, frame, frame_number)
                        if not await self.send_processed_frame(writer, processed_frame):
                            break
                            
                        frame_number += 1
                        if frame_number % 100 == 0:
                            elapsed = time.time() - session_start
                            current_fps = frame_number / elapsed
                            logging.info(f"Processed frames: {frame_number}, FPS: {current_fps:.2f}")
                            
                    except asyncio.CancelledError:
                        logging.info("Client processing cancelled")
                        break
                    except Exception as e:
                        logging.error(f"Frame processing error: {e}")
                        if not writer.is_closing():
                            await writer.drain()
                        else:
                            break
                            
        except Exception as e:
            logging.error(f"Client handler error: {e}")
        finally:
            await self.cleanup_session(writer, session_id)
            
    async def log_session_summary(self, frame_number: int, session_start_time: float):
        """세션 요약 정보 로깅"""
        try:
            total_time = time.time() - session_start_time
            avg_fps = frame_number/total_time if total_time > 0 else 0
            
            logging.info("\nSession Summary:")
            logging.info(f"Total time: {total_time:.2f} seconds")
            logging.info(f"Frames processed: {frame_number}")
            logging.info(f"Average FPS: {avg_fps:.2f}")
            
            # 메모리 사용량
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            logging.info(f"Final memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
            
            # NPU 상태 로깅 추가
            logging.info("Session ended - Cleaning up resources")
            
        except Exception as e:
            logging.error(f"Error logging session summary: {str(e)}")

    
    async def process_frame(self, runner, frame: np.ndarray, frame_number: int) -> np.ndarray:
        try:
            # 프레임 유효성 검사
            if not self.validate_frame(frame):
                logging.error("Invalid frame received")
                return frame

            # 처리 시작 시간 기록 
            process_start = time.time()
            
            try:
                # NPU 추론 수행
                input_tensor, context = self.preprocessor(frame)
                outputs = await runner.run(input_tensor)
                predictions = self.decoder(outputs, context, frame.shape[:2])
                result = frame.copy()
            except Exception as e:
                logging.error(f"NPU inference error: {e}")
                return frame

            if len(predictions) > 0 and len(predictions[0]) > 0:
                # 검출 필터링
                valid_detections = self.filter_detections(predictions, frame, frame.shape)
                
                # 트래킹 업데이트
                await self.update_tracks(valid_detections, frame_number, frame)

                # 활성 트랙 처리
                active_tracks = {
                    track_id: track_info 
                    for track_id, track_info in sorted(
                        self.tracks.items(), 
                        key=lambda x: x[1].get('confidence', 0),
                        reverse=True
                    )[:self.max_active_tracks]
                    if frame_number - track_info['last_seen'] <= 60  # 트래킹 유지 시간 증가
                }

                for track_id, track_info in active_tracks.items():
                    try:
                        # 바운딩 박스 좌표 계산
                        x1, y1, x2, y2 = map(int, track_info['bbox'])
                        
                        # 좌표 범위 검증
                        x1 = max(0, min(x1, frame.shape[1]-1))
                        y1 = max(0, min(y1, frame.shape[0]-1))
                        x2 = max(0, min(x2, frame.shape[1]-1))
                        y2 = max(0, min(y2, frame.shape[0]-1))
                        
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # ROI 추출 및 DB 업데이트
                        face_roi = self.extract_face_roi(frame, x1, y1, x2, y2)
                        if face_roi is not None:
                            self.update_face_db(track_id, face_roi)

                        # 위치 이력 업데이트  
                        self.update_position_history(track_id, [x1, y1, x2, y2], frame.shape)

                        # 모드별 처리
                        is_host = False
                        if self.mosaic_mode == MosaicMode.ALL:
                            is_host = False  # ALL 모드에서는 모두 non-host
                        elif self.mosaic_mode == MosaicMode.NONE:
                            is_host = True
                        elif self.mosaic_mode == MosaicMode.AUTO:
                            if track_id not in self.excluded_ids and track_info.get('confidence', 0) > 0.3:
                                score = float(self.integrate_subject_detection(
                                    track_id, [x1, y1, x2, y2], face_roi,
                                    frame.shape, frame_number
                                ))
                            is_host = track_id in self.excluded_ids
                        else:  # SELECTIVE 모드
                            is_host = track_id in self.excluded_ids

                        # 바운딩 박스 및 라벨 표시  
                        confidence = float(track_info.get('confidence', 0))
                        host_status = 'y' if is_host else 'n'
                        
                        label = f"ID:{track_id} ({confidence:.2f}) host:{host_status}"
                        color = (0, 255, 0) if is_host else (0, 0, 255)
                        
                        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(result, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # 모자이크 처리 - ALL 모드이거나 모자이크 적용 대상인 경우
                        if self.mosaic_mode == MosaicMode.ALL or (not is_host and self.should_apply_mosaic(track_id)):
                            self.apply_mosaic(result, x1, y1, x2, y2)

                    except Exception as e:
                        logging.error(f"Track processing error (ID: {track_id}): {str(e)}")
                        continue

                # 리소스 정리
                if frame_number % 50 == 0:
                    await self.cleanup_resources(frame_number)

            # 성능 메트릭 업데이트
            process_time = time.time() - process_start
            self.performance_metrics['process_times'].append(process_time)
            self.performance_metrics['total_process_time'] += process_time
            self.performance_metrics['frame_count'] = frame_number

            # FPS 로깅
            current_time = time.time()
            if current_time - self.performance_metrics['last_fps_update'] > 5.0:
                elapsed = current_time - self.performance_metrics['start_time']
                avg_fps = self.performance_metrics['frame_count'] / elapsed
                logging.info(f"Performance - FPS: {avg_fps:.2f}, Frames: {self.performance_metrics['frame_count']}")
                self.performance_metrics['last_fps_update'] = current_time

            return result

        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}")
            return frame
        
    async def stream_video(self) -> AsyncGenerator[bytes, None]:
        """비디오 스트리밍 생성기"""
        if not self.cap:
            logging.error("Video capture not initialized")
            return

        last_frame_time = time.time()
        frame_count = 0

        while self.running:
            try:
                current_time = time.time()
                elapsed = current_time - last_frame_time

                if elapsed < self.frame_delay:
                    await asyncio.sleep(self.frame_delay - elapsed)

                ret, frame = self.cap.read()
                if not ret:
                    logging.info("End of video reached, restarting...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # 프레임 처리
                processed_frame = await self.process_frame(frame)
                if processed_frame is None:
                    continue

                # JPEG 인코딩
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                ret, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
                if not ret:
                    logging.warning("Failed to encode frame")
                    continue

                # 프레임 전송
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # 성능 모니터링
                frame_count += 1
                if frame_count % 30 == 0:  # 매 30프레임마다 상태 로깅
                    current_fps = 30 / (time.time() - last_frame_time)
                    logging.info(f"Streaming stats - FPS: {current_fps:.2f}, "
                               f"Frames: {frame_count}")

                last_frame_time = current_time

            except Exception as e:
                logging.error(f"Streaming error: {e}")
                await asyncio.sleep(0.1)
        
    async def handle_client(self, reader, writer):
        """클라이언트 처리"""
        client_address = writer.get_extra_info('peername')
        session_id = id(writer)
        frame_number = 0
        session_start = time.time()
        
        logging.info(f"New client connected from {client_address}")
        
        try:
            # 클라이언트 설정 수신 - writer 인자 추가
            if not await self.receive_client_settings(reader, writer):
                logging.error("Failed to receive client settings")
                return
                
            # 메인 처리 루프
            async with create_runner(self.model_path, device="npu0pe0,npu0pe1,npu1pe0,npu1pe1") as runner:
                while True:
                    try:
                        frame = await self.receive_frame(reader)
                        if frame is None:
                            break
                            
                        processed_frame = await self.process_frame(runner, frame, frame_number)
                        if not await self.send_processed_frame(writer, processed_frame):
                            break
                            
                        frame_number += 1
                        if frame_number % 100 == 0:
                            elapsed = time.time() - session_start
                            current_fps = frame_number / elapsed
                            logging.info(f"Processed frames: {frame_number}, FPS: {current_fps:.2f}")
                            
                    except asyncio.CancelledError:
                        logging.info("Client processing cancelled")
                        break
                    except Exception as e:
                        logging.error(f"Frame processing error: {e}")
                        if not writer.is_closing():
                            await writer.drain()
                        else:
                            break
                            
        except Exception as e:
            logging.error(f"Client handler error: {e}")
        finally:
            await self.cleanup_session(writer, session_id)

    async def start(self):
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port)
        
        print(f"Server running on {self.host}:{self.port}")
        
        async with server:
            await server.serve_forever()
            
    def setup_web_server(self):
        """웹 서버 설정"""
        self.web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.web_app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            return self.templates.TemplateResponse(
                "index.html", 
                {"request": request}
            )
        
        @self.web_app.get("/video_feed")
        async def video_feed():
            return StreamingResponse(
                self.get_web_frame(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
            
        @self.web_app.get("/metrics")
        async def get_metrics():
            """성능 메트릭 API"""
            try:
                if self.performance_metrics:
                    fps = len(self.performance_metrics['process_times']) / (
                        sum(self.performance_metrics['process_times']) 
                        if self.performance_metrics['process_times'] else 1
                    )
                    return {
                        "fps": float(fps),
                        "avg_processing_time": float(
                            sum(self.performance_metrics['process_times']) / 
                            len(self.performance_metrics['process_times'])
                            if self.performance_metrics['process_times'] else 0
                        ),
                        "total_frames": self.performance_metrics['frame_count']
                    }
                return {"error": "Performance metrics not available"}
            except Exception as e:
                logging.error(f"Error getting metrics: {e}")
                return {"error": str(e)}

        @self.web_app.post("/update_mode")
        async def update_mode(request: Request):
            try:
                data = await request.json()
                mode = data.get('mode')
                logging.info(f"Received mode update request: {mode}")

                if mode not in ["all", "none", "selective", "auto"]:
                    return {"status": "error", "message": "Invalid mode"}

                success = self.update_mosaic_settings(mode, [])
                if success:
                    logging.info(f"Mosaic mode updated to: {mode}")
                    return {"status": "success", "mode": mode}
                return {"status": "error", "message": "Failed to update mode"}
            except Exception as e:
                logging.error(f"Failed to update mode: {e}")
                return {"status": "error", "message": str(e)}

        def run_web_server():
            uvicorn.run(self.web_app, host="0.0.0.0", port=8000)
                
        self.web_thread = threading.Thread(target=run_web_server)
        self.web_thread.daemon = True
        self.web_thread.start()

        async def get_web_frame(self):
            """웹 스트리밍용 프레임 생성기"""
            while True:
                if self.latest_frame is not None:
                    try:
                        ret, buffer = cv2.imencode('.jpg', self.latest_frame)
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    except Exception as e:
                        print(f"Frame encoding error: {e}")
                await asyncio.sleep(0.033)  # ~30 FPS

async def main():
    MODEL_PATH = "/home/ubuntu/AI-Semi/contest/models/yolov8s_od.enf"
    SUBJECT_FACE = ""
    
    print("Starting NPU video processing server...")
    server = VideoProcessingServer(MODEL_PATH)
    
    if os.path.exists(SUBJECT_FACE):
        server.add_reference_face(SUBJECT_FACE, "GUMA")
    else:
        print("주체 얼굴 이미지가 없습니다.")
    
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())