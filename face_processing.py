import os
import cv2
import asyncio
import numpy as np
import time
import sqlite3
from datetime import datetime
from furiosa.runtime import create_runner
from utils.preprocess import YOLOPreProcessor
from utils.postprocess_func.output_decoder import ObjDetDecoder

class FaceProcessingSystem:
    def __init__(self, model_path: str, db_path: str = "tracking.db"):
        self.model_path = model_path
        self.preprocessor = YOLOPreProcessor()
        self.decoder = ObjDetDecoder("yolov8n", conf_thres=0.3, iou_thres=0.3)
        self.running = True
        self.total_detections = 0
        self.start_time = None
        self.tracks = {}
        self.next_id = 1
        self.db_path = db_path
        self.init_db()
        print(f"모델 {model_path}로 얼굴 처리 시스템 초기화 완료")

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS detections
                    (id INTEGER PRIMARY KEY,
                     track_id INTEGER,
                     timestamp TEXT,
                     frame_number INTEGER,
                     confidence REAL,
                     x1 INTEGER,
                     y1 INTEGER,
                     x2 INTEGER,
                     y2 INTEGER)''')
        conn.commit()
        conn.close()

    def preprocess_frame(self, frame):
        input_tensor, context = self.preprocessor(frame)
        return input_tensor, context

    def postprocess_frame(self, frame, predictions):
        result = frame.copy()
        if len(predictions) > 0 and len(predictions[0]) > 0:
            prediction = predictions[0]
            valid_detections = []
            
            for det in prediction:
                if det[4] > 0.3 and int(det[5]) == 0:  # person class
                    x1, y1, x2, y2 = map(int, det[:4])
                    
                    # 얼굴 영역 추정
                    person_height = y2 - y1
                    face_height = int(person_height * 0.25)
                    face_y2 = y1 + face_height
                    face_area = result[y1:face_y2, x1:x2]

                    if face_area.size > 0:
                        face_blur = cv2.blur(face_area, (20, 20))
                        result[y1:face_y2, x1:x2] = face_blur
                        
                        self.total_detections += 1

        return result

    async def process_batch(self, frames, runner):
        batch_size = len(frames)
        if batch_size == 0:
            return []

        # 전처리
        input_tensors = []
        contexts = []
        for frame in frames:
            input_tensor, context = self.preprocess_frame(frame)
            input_tensors.append(input_tensor)
            contexts.append(context)

        # 배치 추론
        batch_input = np.concatenate(input_tensors, axis=0)
        batch_outputs = await runner.run(batch_input)

        # 후처리
        processed_frames = []
        for i in range(batch_size):
            predictions = self.decoder([batch_outputs[i]], contexts[i], frames[i].shape[:2])
            processed_frame = self.postprocess_frame(frames[i], predictions)
            processed_frames.append(processed_frame)

        return processed_frames
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS detections
                    (id INTEGER PRIMARY KEY,
                     track_id INTEGER,
                     timestamp TEXT,
                     frame_number INTEGER,
                     confidence REAL,
                     x1 INTEGER,
                     y1 INTEGER,
                     x2 INTEGER,
                     y2 INTEGER)''')
        conn.commit()
        conn.close()

    def calculate_iou(self, box1, box2):
        # Convert to (x1, y1, x2, y2)
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def update_tracks(self, detections, frame_number):
        if not self.tracks:  # 첫 프레임이거나 트랙이 없는 경우
            for det in detections:
                self.tracks[self.next_id] = {
                    'bbox': det[:4],
                    'last_seen': frame_number,
                    'confidence': det[4]
                }
                self.next_id += 1
            return

        # 현재 검출된 박스들과 기존 트랙들 간의 매칭
        matched_tracks = set()
        matched_detections = set()
        
        for track_id, track_info in self.tracks.items():
            if frame_number - track_info['last_seen'] > 30:  # 1초(30프레임) 이상 안 보이면 트랙 삭제
                continue
                
            for i, det in enumerate(detections):
                if i in matched_detections:
                    continue
                    
                iou = self.calculate_iou(track_info['bbox'], det[:4])
                if iou > 0.3:  # IOU 임계값
                    self.tracks[track_id]['bbox'] = det[:4]
                    self.tracks[track_id]['last_seen'] = frame_number
                    self.tracks[track_id]['confidence'] = det[4]
                    matched_tracks.add(track_id)
                    matched_detections.add(i)
                    break
        
        # 매칭되지 않은 검출 결과에 대해 새로운 트랙 생성
        for i, det in enumerate(detections):
            if i not in matched_detections:
                self.tracks[self.next_id] = {
                    'bbox': det[:4],
                    'last_seen': frame_number,
                    'confidence': det[4]
                }
                self.next_id += 1

    def save_to_db(self, track_id, frame_number, detection):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO detections 
                    (track_id, timestamp, frame_number, confidence, x1, y1, x2, y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                 (track_id,
                  datetime.now().isoformat(),
                  frame_number,
                  detection[4],
                  int(detection[0]),
                  int(detection[1]),
                  int(detection[2]),
                  int(detection[3])))
        conn.commit()
        conn.close()

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
                
                # 트래킹 업데이트
                self.update_tracks(valid_detections, frame_number)
                
                # 검출된 사람들에 대해 처리
                for track_id, track_info in self.tracks.items():
                    if frame_number - track_info['last_seen'] > 30:  # 오래된 트랙 스킵
                        continue
                        
                    x1, y1, x2, y2 = map(int, track_info['bbox'])
                    
                    # 얼굴 영역 추정 (상체 25% 정도로 조정)
                    person_height = y2 - y1
                    face_height = int(person_height * 0.25)  # 25%로 조정
                    face_y2 = y1 + face_height
                    face_area = result[y1:face_y2, x1:x2]

                    if face_area.size > 0:
                        # 블러 처리
                        face_blur = cv2.blur(face_area, (20, 20))
                        result[y1:face_y2, x1:x2] = face_blur
                        
                        # ID 표시
                        cv2.putText(result, f"ID: {track_id}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # DB에 저장
                        self.save_to_db(track_id, frame_number, 
                                      [x1, y1, x2, y2, track_info['confidence']])
                        
                        self.total_detections += 1
                        
            return result

        except Exception as e:
            print(f"process_frame 오류: {str(e)}")
            return frame

    async def process_frame_simple(self, frame: np.ndarray, predictions) -> np.ndarray:
        """Simplified frame processing for batch operations"""
        try:
            result = frame.copy()

            if len(predictions) > 0 and len(predictions[0]) > 0:
                prediction = predictions[0]
                
                for det in prediction:
                    if det[4] > 0.3 and int(det[5]) == 0:  # person class
                        x1, y1, x2, y2 = map(int, det[:4])
                        
                        # Face area estimation
                        person_height = y2 - y1
                        face_height = int(person_height * 0.25)
                        face_y2 = y1 + face_height
                        face_area = result[y1:face_y2, x1:x2]

                        if face_area.size > 0:
                            face_blur = cv2.blur(face_area, (20, 20))
                            result[y1:face_y2, x1:x2] = face_blur
                
            return result

        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return frame

    async def process_video(self, video_path: str, output_path: str):
        print(f"\n비디오 처리 중: {video_path}")
        self.start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 열기에 실패했습니다: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"비디오 속성: {width}x{height} @ {fps:.2f}fps, 총 {total_frames} 프레임")

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        async with create_runner(self.model_path, device="npu0pe0,npu0pe1,npu1pe0,npu1pe1") as runner:
            frame_count = 0
            tasks = []

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                task = asyncio.create_task(self.process_frame(runner, frame, frame_count))
                tasks.append(task)

                frame_count += 1

                if frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    progress = (frame_count / total_frames) * 100
                    print(f"\r진행률: {progress:.1f}% | FPS: {current_fps:.2f} | 탐지 수: {self.total_detections}",
                            end='', flush=True)

            results = await asyncio.gather(*tasks)

            for processed_frame in results:
                out.write(processed_frame)

        cap.release()
        out.release()

        total_time = time.time() - self.start_time
        print(f"\n\n처리 완료!")
        print(f"총 소요 시간: {total_time:.2f} 초")
        print(f"평균 FPS: {total_frames / total_time:.2f}")
        print(f"총 탐지 수: {self.total_detections}")

async def main():
    MODEL_PATH = "/home/ubuntu/AI-Semi/contest/models/yolov8n_single.enf"
    VIDEO_PATH = "/home/ubuntu/AI-Semi/contest/dataset/video/남순영상3.mp4"
    OUTPUT_PATH = "/home/ubuntu/AI-Semi/contest/dataset/output/남순영상3_1754.mp4"

    print("얼굴 처리 시스템 초기화 중...")
    processor = FaceProcessingSystem(MODEL_PATH)
    await processor.process_video(VIDEO_PATH, OUTPUT_PATH)

if __name__ == "__main__":
    asyncio.run(main())
