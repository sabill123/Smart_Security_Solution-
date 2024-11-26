import cv2
import asyncio
import numpy as np
import time
import struct
import pickle
import argparse
from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor

class VideoClient:
    def __init__(self, host: str = 'localhost', port: int = 8485):
        self.host = host
        self.port = port
        self.frame_count = 0
        self.start_time = None
        self.reader = None
        self.writer = None
        self.frame_buffer_size = 4  # 프레임 버퍼 크기
        self.executor = ThreadPoolExecutor(max_workers=2)  # 인코딩/디코딩용 스레드 풀

    async def setup_connection(self):
        try:
            self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
            print(f"Connected to server at {self.host}:{self.port}")
            
            # 모자이크 설정 전송
            settings = {
                'mosaic_mode': self.mosaic_mode.value,
                'excluded_ids': list(self.excluded_ids)
            }
            settings_data = pickle.dumps(settings)
            self.writer.write(struct.pack('Q', len(settings_data)))
            self.writer.write(settings_data)
            await self.writer.drain()
            
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def process_stream(self, source: Union[str, int], output_path: Optional[str] = None):
        if isinstance(source, str) and source.startswith('rtsp'):
            cap = cv2.VideoCapture(source)
            stream_type = 'rtsp'
        elif isinstance(source, int) or source == 'webcam':
            cap = cv2.VideoCapture(0 if source == 'webcam' else source)
            stream_type = 'webcam'
        else:
            cap = cv2.VideoCapture(source)
            stream_type = 'file'

        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nVideo Information:")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        print(f"Total Frames: {total_frames}")
        print(f"Estimated Duration: {total_frames/fps:.2f} seconds")
        print("\nStarting processing...")

        if not await self.setup_connection():
            return

        self.writer.write(struct.pack('2i', width, height))
        await self.writer.drain()

        out = None
        if output_path:
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )

        self.start_time = time.time()
        last_fps_print = time.time()
        reconnect_attempts = 0
        max_reconnect_attempts = 5

        # 프레임 버퍼링을 위한 큐
        frame_queue = asyncio.Queue(maxsize=self.frame_buffer_size)
        result_queue = asyncio.Queue()

        # 프레임 전송 및 수신 태스크
        send_task = asyncio.create_task(self._send_frames(frame_queue))
        receive_task = asyncio.create_task(self._receive_frames(result_queue))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if stream_type in ['rtsp', 'webcam']:
                        print("\nStream interrupted, attempting to reconnect...")
                        await asyncio.sleep(1)
                        cap.release()
                        cap = cv2.VideoCapture(source)
                        if not cap.isOpened():
                            reconnect_attempts += 1
                            if reconnect_attempts >= max_reconnect_attempts:
                                print("Max reconnection attempts reached")
                                break
                            continue
                        reconnect_attempts = 0
                        continue
                    else:
                        print("\nEnd of video file")
                        break

                # 프레임을 전송 큐에 추가
                await frame_queue.put(frame)

                try:
                    # 처리된 프레임 수신
                    processed_frame = await result_queue.get()
                    self.frame_count += 1

                    if out:
                        out.write(processed_frame)
                    
                    if not output_path or stream_type in ['rtsp', 'webcam']:
                        cv2.imshow('Processed Stream', processed_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("\nStopped by user")
                            break
                        elif key == ord('s') and stream_type in ['rtsp', 'webcam']:
                            snapshot_path = f"snapshot_{int(time.time())}.jpg"
                            cv2.imwrite(snapshot_path, processed_frame)
                            print(f"\nSnapshot saved: {snapshot_path}")

                    current_time = time.time()
                    if current_time - last_fps_print >= 1.0:
                        elapsed = current_time - self.start_time
                        fps = self.frame_count / elapsed
                        progress = (self.frame_count / total_frames) * 100 if total_frames > 0 else 0
                        estimated_time_left = (total_frames - self.frame_count) / fps if fps > 0 and total_frames > 0 else 0

                        print(f"\rProgress: {progress:.1f}% | "
                              f"Frame: {self.frame_count}/{total_frames} | "
                              f"FPS: {fps:.2f} | "
                              f"Elapsed: {elapsed:.1f}s | "
                              f"ETA: {estimated_time_left:.1f}s",
                              end='', flush=True)
                        last_fps_print = current_time

                except asyncio.QueueEmpty:
                    continue

        except Exception as e:
            print(f"\nProcessing error: {e}")
        finally:
            # 정리
            send_task.cancel()
            receive_task.cancel()
            try:
                await send_task
                await receive_task
            except asyncio.CancelledError:
                pass

            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            self.executor.shutdown()

            if self.start_time:
                total_time = time.time() - self.start_time
                print(f"\n\nProcessing completed!")
                print(f"Total time: {total_time:.2f} seconds")
                print(f"Total frames: {self.frame_count}")
                print(f"Average FPS: {self.frame_count/total_time:.2f}")

    async def _send_frames(self, frame_queue):
        try:
            while True:
                frame = await frame_queue.get()
                if frame is None:
                    break
                    
                frame_data = await asyncio.get_event_loop().run_in_executor(
                    self.executor, pickle.dumps, frame)
                
                data_size = len(frame_data)
                if data_size > 0:  # 데이터 크기 확인
                    self.writer.write(struct.pack('Q', data_size))
                    self.writer.write(frame_data)
                    await self.writer.drain()
                else:
                    print(f"Warning: Invalid frame data size: {data_size}")
                    
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"Send frames error: {e}")

    async def _receive_frames(self, result_queue):
        """처리된 프레임 수신을 위한 백그라운드 태스크"""
        try:
            while True:
                size_data = await self.reader.readexactly(struct.calcsize('Q'))
                frame_size = struct.unpack('Q', size_data)[0]
                frame_data = await self.reader.readexactly(frame_size)
                
                # 비동기 역직렬화
                processed_frame = await asyncio.get_event_loop().run_in_executor(
                    self.executor, pickle.loads, frame_data)
                
                await result_queue.put(processed_frame)
        except asyncio.CancelledError:
            return

    async def reconnect(self):
        print("Attempting to reconnect to server...")
        try:
            await self.setup_connection()
            return True
        except:
            print("Reconnection failed")
            return False

def main():
    parser = argparse.ArgumentParser(description='Video Processing Client')
    parser.add_argument('--source', type=str, 
                       default='/home/ubuntu/AI-Semi/warboy_tutorial/part5/dataset/video/조두순1.mp4',
                       help='Video source (file path, webcam, or rtsp url)')
    parser.add_argument('--output', type=str, 
                       default='/home/ubuntu/AI-Semi/warboy_tutorial/part5/dataset/output/조두순1_1605.mp4',
                       help='Output video path (optional)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Server host')
    parser.add_argument('--port', type=int, default=8485,
                       help='Server port')

    args = parser.parse_args()

    client = VideoClient(args.host, args.port)
    
    try:
        asyncio.run(client.process_stream(args.source, args.output))
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()