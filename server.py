import os
import cv2
import asyncio
import numpy as np
import time
from furiosa.runtime import create_runner
from utils.preprocess import YOLOPreProcessor
from utils.postprocess_func.output_decoder import ObjDetDecoder
import pickle
import struct
from collections import defaultdict

class VideoProcessingServer:
    def __init__(self, model_path: str, host: str = '0.0.0.0', port: int = 8485, reference_face: str = None):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.preprocessor = YOLOPreProcessor()
        self.decoder = ObjDetDecoder("yolov8n", conf_thres=0.3, iou_thres=0.3)
        self.running = True
        self.total_detections = 0
        self.start_time = None
        self.tracks = {}
        self.next_id = 1
        
        # 주체 처리 관련 변수들
        self.excluded_ids = set()
        self.reference_face = None
        self.first_frame_processed = False
        self.position_history = defaultdict(list)
        self.frame_presence = defaultdict(int)
        self.reference_faces = {}  # 주체별 얼굴 이미지 저장
        self.face_features = {}    # 얼굴 특징 저장
        
        # 주체 판단 기준값
        self.MIN_FRAMES = 30
        self.MIN_SIZE_RATIO = 0.25
        self.MAX_MOVEMENT = 0.2
        
        if reference_face and os.path.exists(reference_face):
            self.reference_face = cv2.imread(reference_face)
            print("기준 얼굴 이미지 로드됨")
        
        print(f"서버 초기화 완료 - 모델: {model_path}")
        
    def add_reference_face(self, face_image_path: str, label: str = "main_subject"):
        """주체의 얼굴 이미지 등록"""
        if os.path.exists(face_image_path):
            face = cv2.imread(face_image_path)
            if face is not None:
                face_gray = cv2.cvtColor(cv2.resize(face, (64, 64)), cv2.COLOR_BGR2GRAY)
                self.reference_faces[label] = face
                self.face_features[label] = face_gray
                print(f"주체 얼굴 등록 완료: {label}")
                return True
        return False

    def calculate_face_similarity(self, face1, face2):
        if face1 is None or face2 is None:
            return 0
        
        try:
            face1 = cv2.resize(face1, (64, 64))
            face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
            
            max_similarity = 0
            for label, ref_face_gray in self.face_features.items():
                similarity = cv2.matchTemplate(face1_gray, ref_face_gray, cv2.TM_CCOEFF_NORMED)[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    
            return max_similarity
        except:
            return 0

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        return intersection / union if union > 0 else 0

    def update_position_history(self, track_id, bbox, frame_shape):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2 / frame_shape[1]
        size_ratio = (x2 - x1) * (y2 - y1) / (frame_shape[0] * frame_shape[1])
        
        self.position_history[track_id].append((center_x, size_ratio))
        self.frame_presence[track_id] += 1
        
        if len(self.position_history[track_id]) > self.MIN_FRAMES:
            self.position_history[track_id].pop(0)
            
        if len(self.position_history[track_id]) >= self.MIN_FRAMES:
            positions = [p[0] for p in self.position_history[track_id]]
            sizes = [p[1] for p in self.position_history[track_id]]
            
            movement_range = max(positions) - min(positions)
            avg_size = sum(sizes) / len(sizes)
            
            if (movement_range < self.MAX_MOVEMENT and 
                avg_size > self.MIN_SIZE_RATIO and 
                track_id not in self.excluded_ids):
                self.excluded_ids.add(track_id)
                print(f"위치 기반 주체 설정: ID {track_id}")

    def update_tracks(self, detections, frame_number):
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = {
                    'bbox': det[:4],
                    'last_seen': frame_number,
                    'confidence': det[4]
                }
                self.next_id += 1
            return

        matched_tracks = set()
        matched_detections = set()
        
        for track_id, track_info in list(self.tracks.items()):
            if frame_number - track_info['last_seen'] > 30:
                continue
                
            for i, det in enumerate(detections):
                if i in matched_detections:
                    continue
                    
                iou = self.calculate_iou(track_info['bbox'], det[:4])
                if iou > 0.3:
                    self.tracks[track_id]['bbox'] = det[:4]
                    self.tracks[track_id]['last_seen'] = frame_number
                    self.tracks[track_id]['confidence'] = det[4]
                    matched_tracks.add(track_id)
                    matched_detections.add(i)
                    break

        for i, det in enumerate(detections):
            if i not in matched_detections:
                self.tracks[self.next_id] = {
                    'bbox': det[:4],
                    'last_seen': frame_number,
                    'confidence': det[4]
                }
                self.next_id += 1

    def extract_face_roi(self, frame, x1, y1, x2, y2):
        person_height = y2 - y1
        head_height = int(person_height * 0.25)
        head_width = int((x2 - x1) * 0.6)
        head_x1 = x1 + ((x2 - x1) - head_width) // 2
        head_x2 = head_x1 + head_width
        head_y2 = min(y1 + head_height, frame.shape[0])
        
        if head_x2 > head_x1 and head_y2 > y1:
            return frame[y1:head_y2, head_x1:head_x2]
        return None

    def find_subject_by_size(self, detections):
        if not detections:
            return None
        
        max_area = 0
        subject_detection = None
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                subject_detection = det
                
        return subject_detection

    def calculate_face_similarity(self, face1, face2):
        if face1 is None or face2 is None:
            return 0
            
        try:
            face1 = cv2.resize(face1, (64, 64))
            face2 = cv2.resize(face2, (64, 64))
            
            face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
            face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
            
            similarity = cv2.matchTemplate(face1_gray, face2_gray, cv2.TM_CCOEFF_NORMED)
            return similarity[0][0]
        except:
            return 0

    async def process_frame(self, runner, frame: np.ndarray, frame_number: int) -> np.ndarray:
        try:
            input_tensor, context = self.preprocessor(frame)
            outputs = await runner.run(input_tensor)
            predictions = self.decoder(outputs, context, frame.shape[:2])
            result = frame.copy()

            if len(predictions) > 0 and len(predictions[0]) > 0:
                prediction = predictions[0]
                valid_detections = []
                
                for det in prediction:
                    if det[4] > 0.3 and int(det[5]) == 0:  # person class
                        valid_detections.append(det)

                # 첫 프레임에서 주체 설정
                if not self.first_frame_processed:
                    subject = self.find_subject_by_size(valid_detections)
                    if subject is not None:
                        self.excluded_ids.add(self.next_id)
                        print(f"크기 기반 주체 설정: ID {self.next_id}")
                    self.first_frame_processed = True
                
                self.update_tracks(valid_detections, frame_number)
                
                for track_id, track_info in self.tracks.items():
                    if frame_number - track_info['last_seen'] > 30:
                        continue
                        
                    x1, y1, x2, y2 = map(int, track_info['bbox'])
                    
                    # 위치 기반 주체 판단
                    self.update_position_history(track_id, [x1, y1, x2, y2], frame.shape)
                    
                    # 얼굴 유사도 체크
                    face_roi = self.extract_face_roi(frame, x1, y1, x2, y2)
                    if face_roi is not None and self.calculate_face_similarity(face_roi, None) > 0.7:
                        self.excluded_ids.add(track_id)
                        # 동일 인물로 판단된 ID들도 제외 목록에 추가
                        for other_id in list(self.tracks.keys()):
                            if other_id != track_id and self.are_same_person(track_id, other_id):
                                self.excluded_ids.add(other_id)
                                print(f"동일 인물 ID 추가: {other_id}")

                    # 모자이크 적용 (주체 제외)
                    if track_id not in self.excluded_ids:
                        person_height = y2 - y1
                        relative_size = person_height / frame.shape[0]
                        
                        if relative_size > 0.4:
                            head_height = int(person_height * 0.35)
                            head_width_ratio = 0.7
                            kernel_size = 31
                        elif relative_size > 0.25:
                            head_height = int(person_height * 0.25)
                            head_width_ratio = 0.6
                            kernel_size = 25
                        else:
                            head_height = int(person_height * 0.2)
                            head_width_ratio = 0.5
                            kernel_size = 21

                        head_y2 = min(y1 + head_height, frame.shape[0])
                        head_width = int((x2 - x1) * head_width_ratio)
                        head_x1 = x1 + ((x2 - x1) - head_width) // 2
                        head_x2 = head_x1 + head_width

                        head_x1 = max(0, head_x1)
                        head_x2 = min(frame.shape[1], head_x2)
                        head_y1 = max(0, y1)
                        head_y2 = min(frame.shape[0], head_y2)

                        if head_x2 > head_x1 and head_y2 > head_y1:
                            mask = np.zeros((head_y2 - head_y1, head_x2 - head_x1), dtype=np.uint8)
                            center = ((head_x2 - head_x1) // 2, (head_y2 - head_y1) // 2)
                            axes = (int((head_x2 - head_x1) * 0.45), int((head_y2 - head_y1) * 0.45))
                            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                            
                            mask = cv2.GaussianBlur(mask, (31, 31), 10)
                            roi = result[head_y1:head_y2, head_x1:head_x2].copy()
                            
                            h, w = roi.shape[:2]
                            small = cv2.resize(roi, (w//8, h//8))
                            blurred = cv2.resize(small, (w, h))
                            blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), 15)

                            mask_3d = mask.reshape(mask.shape + (1,)) / 255.0
                            result[head_y1:head_y2, head_x1:head_x2] = (blurred * mask_3d + 
                                roi * (1 - mask_3d)).astype(np.uint8)

                    # ID와 주체 여부 표시
                    color = (0, 255, 0) if track_id in self.excluded_ids else (0, 0, 255)
                    label = f"ID: {track_id} (주체)" if track_id in self.excluded_ids else f"ID: {track_id}"
                    cv2.putText(result, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    self.total_detections += 1

            return result

        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return frame

    async def handle_client(self, reader, writer):
        print("New client connected!")
        frame_number = 0
        
        try:
            data = await reader.readexactly(8)
            width, height = struct.unpack('2i', data)
            print(f"Video properties: {width}x{height}")
            
            self.start_time = time.time()
            
            async with create_runner(self.model_path, device="npu0pe0,npu0pe1,npu1pe0,npu1pe1") as runner:
                while True:
                    try:
                        size_data = await reader.readexactly(struct.calcsize('Q'))
                    except asyncio.IncompleteReadError:
                        break
                        
                    frame_size = struct.unpack('Q', size_data)[0]
                    
                    frame_data = await reader.readexactly(frame_size)
                    frame = pickle.loads(frame_data)
                    
                    processed_frame = await self.process_frame(runner, frame, frame_number)
                    frame_number += 1
                    
                    processed_data = pickle.dumps(processed_frame)
                    writer.write(struct.pack('Q', len(processed_data)))
                    writer.write(processed_data)
                    await writer.drain()
                    
                    if frame_number % 10 == 0:
                        elapsed = time.time() - self.start_time
                        fps = frame_number / elapsed
                        print(f"\rProcessing FPS: {fps:.2f} | Total detections: {self.total_detections}", 
                                end='', flush=True)
                                
        except Exception as e:
            print(f"\nClient handler error: {str(e)}")
        finally:
            writer.close()
            await writer.wait_closed()
            print("\nClient disconnected")
            
            if self.start_time:
                total_time = time.time() - self.start_time
                print(f"\nSession summary:")
                print(f"Total time: {total_time:.2f} seconds")
                print(f"Frames processed: {frame_number}")
                print(f"Average FPS: {frame_number/total_time:.2f}")
                print(f"Total detections: {self.total_detections}")

    async def start(self):
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port)
        
        print(f"Server running on {self.host}:{self.port}")
        
        async with server:
            await server.serve_forever()

async def main():
    MODEL_PATH = "/home/ubuntu/AI-Semi/contest/models/yolov8n_single.enf"
    
    server = VideoProcessingServer(MODEL_PATH)
    
    # 주체 얼굴 등록
    server.add_reference_face("subject_face.jpg", "main_subject")
    
    await server.start()
if __name__ == "__main__":
    asyncio.run(main())