#client_web.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import argparse

from contextlib import asynccontextmanager
import uvicorn
import asyncio
import cv2
import numpy as np
import socket
import struct
import pickle
import httpx

from pathlib import Path
import time
from typing import Optional, Dict, AsyncIterator,Deque, List
import logging
from datetime import datetime
import os
from collections import deque  # deque import 추가


<<<<<<< HEAD
VIDEO_PATH = "/home/ubuntu/AI-Semi/contest/dataset/video/남순영상3.mp4"
=======
VIDEO_PATH = "/home/ubuntu/AI-Semi/contest/dataset/video/손흥민1.mp4"
>>>>>>> 73c528f1247d8e1d77b7a47b16c39b4b093a9773

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('client.log'),
        logging.StreamHandler()
    ]
)

class PerformanceMonitor:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times: list[float] = []
        self.start_time = time.time()
        self.frame_count = 0
        self.last_log_time = time.time()
        self.processing_times: list[float] = deque(maxlen=window_size)
        
    def update(self, processing_time: float) -> None:
        """성능 메트릭 업데이트"""
        current_time = time.time()
        self.frame_count += 1
        
        self.frame_times.append(current_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
            
        self.processing_times.append(processing_time)
        
        if current_time - self.last_log_time > 10:
            self.log_metrics()
            self.last_log_time = current_time
    
    def get_fps(self) -> float:
        """현재 FPS 계산"""
        if len(self.frame_times) < 2:
            return 0.0
        
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff <= 0:
            return 0.0
            
        return len(self.frame_times) / time_diff
        
    def get_average_processing_time(self) -> float:
        """평균 처리 시간 계산 (초 단위)"""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
        
    def log_metrics(self) -> None:
        """성능 메트릭 로깅"""
        fps = self.get_fps()
        avg_processing_time = self.get_average_processing_time() * 1000  # ms로 변환
        total_time = time.time() - self.start_time
        logging.info(
            f"Performance Metrics - "
            f"FPS: {fps:.2f}, "
            f"Avg Processing Time: {avg_processing_time:.2f}ms, "
            f"Total Frames: {self.frame_count}, "
            f"Runtime: {total_time:.2f}s"
        )

class VideoClient:
    def __init__(self, host: str = 'localhost', port: int = 8485):
        self.host = host
        self.port = port
        self.initial_mosaic_mode = 'auto'  # 기본값
        self.current_mode = 'auto'
        self.selected_ids = set()
        self.sock: Optional[socket.socket] = None
        self.performance_monitor = PerformanceMonitor()
        self.running = True
        self.connected = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.width = None
        self.height = None
        self.fps = None
        logging.info(f"VideoClient initialized with {host}:{port}")
        
    def send_initial_settings(self):
        """초기 설정 전송"""
        try:
            settings = {
                'mosaic_mode': 'auto',
                'excluded_ids': []
            }
            settings_data = pickle.dumps(settings)
            size = len(settings_data)
            
            # 크기 전송
            self.sock.send(struct.pack('Q', size))
            # 데이터 전송
            self.sock.send(settings_data)
            
            logging.info("Initial settings sent successfully")
        except Exception as e:
            logging.error(f"Failed to send initial settings: {e}")
            raise

    async def initialize(self, video_path: str) -> bool:
        """비디오 클라이언트 초기화"""
        try:
            # 비디오 캡처 초기화
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                logging.error(f"Failed to open video: {video_path}")
                return False
                
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_delay = 1.0 / self.fps
            
            logging.info(f"Video initialized: {self.width}x{self.height} @ {self.fps}fps")

            # 서버 연결 확인
            if not await self.ensure_connection():
                return False

            # 설정 전송
            try:
                settings_result = await self.send_settings()
                if not settings_result:
                    logging.error("Server rejected settings")
                    return False
                return True
            except Exception as e:
                logging.error(f"Failed to send initial settings: {e}")
                return False

        except Exception as e:
            logging.error(f"Initialization error: {e}")
            return False

    async def connect_and_initialize(self) -> bool:
        """서버 연결 및 초기 설정"""
        try:
            if not self.connected:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                self.connected = True
                logging.info(f"Successfully connected to server at {self.host}:{self.port}")

            # 비디오 정보를 포함한 설정 전송
            settings = {
                'mosaic_mode': 'auto',
                'excluded_ids': [],
                'video_info': {
                    'width': self.width,
                    'height': self.height,
                    'fps': self.fps
                }
            }
            
            settings_data = pickle.dumps(settings)
            size = len(settings_data)
            
            self.sock.send(struct.pack('Q', size))
            self.sock.send(settings_data)
            
            # 비디오 크기 정보 전송
            self.sock.send(struct.pack('2i', self.width, self.height))
            
            logging.info("Connection and initialization successful")
            return True
            
        except Exception as e:
            logging.error(f"Connection error: {e}")
            self.connected = False
            if self.sock:
                self.sock.close()
                self.sock = None
            return False

    async def ensure_connection(self) -> bool:
        """연결 상태 확인 및 재연결"""
        try:
            if self.sock is None or not self.connected:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5.0)  # 5초 타임아웃
                self.sock.connect((self.host, self.port))
                self.connected = True
                logging.info(f"Connected to server at {self.host}:{self.port}")
            return True
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            self.connected = False
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
            return False
    
    async def initialize(self, video_path: str) -> bool:
        """비디오 클라이언트 초기화"""
        try:
            # 비디오 캡처 초기화
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                logging.error(f"Failed to open video: {video_path}")
                return False
                
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_delay = 1.0 / self.fps
            
            logging.info(f"Video initialized: {self.width}x{self.height} @ {self.fps}fps")

            # 서버 연결 확인
            if not await self.ensure_connection():
                return False

            # 설정 전송
            try:
                await self.send_settings()
                return True
            except Exception as e:
                logging.error(f"Failed to send initial settings: {e}")
                return False

        except Exception as e:
            logging.error(f"Initialization error: {e}")
            return False

    async def send_settings(self):
        """서버에 설정 전송"""
        if not await self.ensure_connection():
            raise ConnectionError("Not connected to server")

        try:
            # 설정 데이터 준비
            settings = {
                'mosaic_mode': self.initial_mosaic_mode,  # 초기 모드 사용
                'excluded_ids': [],
                'video_info': {
                    'width': self.width,
                    'height': self.height,
                    'fps': self.fps
                }
            }
            
            settings_data = pickle.dumps(settings)
            size = len(settings_data)
            
            # 크기 전송
            self.sock.settimeout(5.0)  # 타임아웃 설정
            self.sock.sendall(struct.pack('Q', size))
            
            # 데이터 전송
            self.sock.sendall(settings_data)
            
            # 비디오 크기 정보 전송
            self.sock.sendall(struct.pack('2i', self.width, self.height))
            
            # 서버 응답 대기
            try:
                response = self.sock.recv(struct.calcsize('i'))
                if response:
                    result = struct.unpack('i', response)[0]
                    if result == 1:
                        logging.info("Server acknowledged settings")
                        return True
                    else:
                        logging.warning("Server rejected settings")
                        return False
                else:
                    logging.warning("No response from server")
                    return False
            except socket.timeout:
                logging.warning("Timeout waiting for server response")
                return False
            finally:
                self.sock.settimeout(None)  # 타임아웃 해제
                
        except Exception as e:
            logging.error(f"Failed to send settings: {e}")
            self.connected = False
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
            raise
        
    async def send_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """프레임 전송 및 응답 수신"""
        if not await self.ensure_connection():
            return None

        try:
            # JPEG으로 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, frame_data = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = frame_data.tobytes()

            # 크기 전송
            size = len(frame_bytes)
            self.sock.sendall(struct.pack('Q', size))
            
            # 데이터 전송
            self.sock.sendall(frame_bytes)

            # 응답 수신
            size_data = self.sock.recv(struct.calcsize('Q'))
            if not size_data:
                raise ConnectionError("Server closed connection")
                
            size = struct.unpack('Q', size_data)[0]
            
            data = bytearray()
            while len(data) < size:
                packet = self.sock.recv(min(size - len(data), 4096))
                if not packet:
                    raise ConnectionError("Connection lost during data reception")
                data.extend(packet)

            # JPEG 디코딩
            nparr = np.frombuffer(data, np.uint8)
            decoded_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if decoded_frame is None:
                raise ValueError("Failed to decode processed frame")
                
            return decoded_frame
            
        except Exception as e:
            logging.error(f"Frame sending error: {e}")
            self.connected = False
            return None

    async def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """프레임 처리"""
        try:
            start_time = time.time()
            
            if self.sock is None:
                logging.error("No server connection")
                return None

            # 프레임 전송 및 처리
            processed_frame = await self.send_frame(frame)
            
            # 성능 메트릭 업데이트
            process_time = time.time() - start_time
            self.performance_monitor.update(process_time)
            
            return processed_frame
            
        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}")
            return None

    async def stream_video(self) -> AsyncIterator[bytes]:
        """비디오 스트리밍"""
        if not self.cap:
            logging.error("Video capture not initialized")
            return
            
        last_frame_time = time.time()
        frame_count = 0
        
        while self.running:
            try:
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                # FPS 제어
                if elapsed < self.frame_delay:
                    await asyncio.sleep(self.frame_delay - elapsed)
                
                ret, frame = self.cap.read()
                if not ret:
                    logging.info("End of video reached, restarting...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                try:
                    # 프레임 처리 및 전송
                    processed_frame = frame  # 기본값은 원본 프레임
                    if self.connected:
                        try:
                            processed_frame = await self.send_frame(frame)
                            if processed_frame is None:
                                processed_frame = frame
                        except Exception as e:
                            logging.error(f"Frame processing error: {e}")
                            # 연결 끊김 처리
                            self.connected = False
                            await self.ensure_connection()
                    
                    # JPEG 인코딩
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    ret, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
                    if not ret:
                        logging.warning("Failed to encode frame")
                        continue
                    
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        current_fps = 30 / (time.time() - last_frame_time)
<<<<<<< HEAD
=======
                        logging.info(f"Streaming FPS: {current_fps:.2f}")
>>>>>>> 73c528f1247d8e1d77b7a47b16c39b4b093a9773
                        last_frame_time = current_time
                        
                except Exception as e:
                    logging.error(f"Frame processing error: {e}")
                    await asyncio.sleep(0.1)
                    continue
                
            except Exception as e:
                logging.error(f"Streaming error: {e}")
                await asyncio.sleep(0.1)
                
    async def update_mosaic_mode(self, mode: str) -> bool:
        """모자이크 모드 업데이트"""
        try:
            if not await self.ensure_connection():
                logging.error("Server connection failed")
                return False

            logging.info(f"Updating mosaic mode to: {mode}")
            settings = {
                'mosaic_mode': mode,
                'excluded_ids': [],
                'video_info': {
                    'width': self.width,
                    'height': self.height,
                    'fps': self.fps
                }
            }
            
            settings_data = pickle.dumps(settings)
            size = len(settings_data)
            
            self.sock.settimeout(5.0)
            try:
                # 설정 전송
                self.sock.sendall(struct.pack('Q', size))
                self.sock.sendall(settings_data)
                self.sock.sendall(struct.pack('2i', self.width, self.height))
                
                # 응답 대기
                response = self.sock.recv(struct.calcsize('i'))
                if response:
                    result = struct.unpack('i', response)[0]
                    if result == 1:
                        logging.info(f"Server acknowledged mode change to: {mode}")
                        return True
                
                logging.warning("Server rejected mode change")
                return False
                
            except socket.timeout:
                logging.error("Timeout while updating mode")
                return False
            finally:
                self.sock.settimeout(None)
                
        except Exception as e:
            logging.error(f"Failed to update mosaic mode: {e}")
            return False
    
    def close(self):
        """리소스 정리"""
        self.running = False
        self.connected = False
        if self.cap:
            self.cap.release()
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            self.sock.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    global video_client
    logging.info("Initializing video client...")
    video_client = VideoClient()
    success = await video_client.initialize(VIDEO_PATH)
    if not success:
        logging.error("Failed to initialize video client")
    
    yield
    
    # 종료 시 실행
    if video_client:
        video_client.close()
        logging.info("Video client closed")

# FastAPI 앱 설정
app = FastAPI()
templates = Jinja2Templates(directory="templates")
video_client: Optional[VideoClient] = None

# Static 파일 설정
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_client
    logging.info("Initializing video client...")
    video_client = VideoClient()
    success = await video_client.initialize(VIDEO_PATH)
    if not success:
        logging.error("Failed to initialize video client")
    
    yield
    
    if video_client:
        video_client.close()
        logging.info("Video client closed")

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    """비디오 스트림 엔드포인트"""
    if video_client is None:
        return HTMLResponse(content="Video client not initialized", status_code=500)
        
    return StreamingResponse(
        video_client.stream_video(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# FastAPI 라우트 추가
@app.post("/update_mode/{mode}")
async def update_mode(mode: str, selected_ids: List[int] = None):
    try:
        if video_client is None:
            return {"status": "error", "message": "Video client not initialized"}

        if mode not in ["all", "none", "selective", "auto"]:
            return {"status": "error", "message": "Invalid mode"}

        success = await video_client.update_mosaic_mode(mode, selected_ids)
        if success:
            return {"status": "success", "mode": mode}
        return {"status": "error", "message": "Failed to update mode"}

    except Exception as e:
        logging.error(f"Mode update error: {e}")
        return {"status": "error", "message": str(e)}
    
@app.post("/update_mode")
async def update_mode(request: Request):
    try:
        data = await request.json()
        mode = data.get('mode')
        selected_ids = data.get('selected_ids')  # 추가
        
        logging.info(f"모드 업데이트 요청: {mode}, 선택된 ID: {selected_ids}")

        if mode not in ["all", "none", "selective", "auto"]:
            return {"status": "error", "message": "잘못된 모드"}

        # 서버로 전달할 데이터에 selected_ids 포함
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/update_mode",
                json={
                    "mode": mode,
                    "selected_ids": selected_ids
                },
                headers={'Content-Type': 'application/json'}
            )
            return response.json()
            
    except Exception as e:
        logging.error(f"모드 업데이트 오류: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/metrics")
async def get_metrics():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/metrics")
            return response.json()
    except Exception as e:
        logging.error(f"Metrics error: {e}")
        return {"error": f"Failed to get metrics: {str(e)}"}

@app.post("/restart")
async def restart_stream():
    try:
        logging.info("Received stream restart request")
        if video_client is None:
            logging.error("Video client not initialized")
            return {"status": "error", "message": "Video client not initialized"}

        video_client.running = False
        await asyncio.sleep(1)
        video_client.running = True
        
        if video_client.cap:
            video_client.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        logging.info("Stream restart successful")
        return {"status": "success", "message": "Stream restarted"}
    except Exception as e:
        logging.error(f"Stream restart error: {e}")
        return {"status": "error", "message": str(e)}


# main 실행 전에 추가
def parse_args():
    parser = argparse.ArgumentParser(description='Video Client')
    parser.add_argument('--mosaic-mode', type=str, default='auto',
                       choices=['all', 'none', 'selective', 'auto'],
                       help='Mosaic mode (default: auto)')
    parser.add_argument('--video-path', type=str, default=VIDEO_PATH,
                       help='Path to video file')
    return parser.parse_args()

if __name__ == "__main__":
<<<<<<< HEAD
    import uvicorn
    import asyncio
    from pathlib import Path

    # 템플릿과 정적 파일 디렉토리 확인
    templates_dir = Path("templates")
    static_dir = Path("static")
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)

    if not (templates_dir / "index.html").exists():
        print("Error: index.html not found in templates directory")
        exit(1)

    logging.info(f"Starting video client with video: {VIDEO_PATH}")
    
    # FastAPI 실행
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
=======
    logging.basicConfig(level=logging.INFO)
    
    args = parse_args()
    logging.info(f"Starting video client with video: {args.video_path}")
    
    # templates 디렉토리 생성 및 확인
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    if not (templates_dir / "index.html").exists():
        logging.error("index.html not found in templates directory")
        raise FileNotFoundError("index.html not found in templates directory")
    
    # FastAPI 실행
    config = uvicorn.Config(
        "client_web:app",
        host="0.0.0.0",
        port=8001,
        reload=True,  # 개발 중 자동 리로드 활성화
        reload_dirs=["templates"],  # 템플릿 디렉토리 변경 감지
        log_level="info"
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
>>>>>>> 73c528f1247d8e1d77b7a47b16c39b4b093a9773
