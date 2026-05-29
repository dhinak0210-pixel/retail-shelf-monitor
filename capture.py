"""
capture.py - Video frame capture module
Supports: webcam (0), RTSP streams, video files, and high-fidelity synthetic demo mode.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Union, Iterator, Optional, Callable
import threading
import queue
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("capture")

class FrameCapture:
    """
    High-performance frame capture with buffering and auto-reconnect.
    
    Usage:
        cap = FrameCapture(source=0)  # Webcam
        cap = FrameCapture(source="rtsp://camera-ip/stream")  # IP camera
        cap = FrameCapture(source="video.mp4")  # File
        cap = FrameCapture(source="synthetic")  # High-fidelity Interactive Demo
        
        for frame in cap.stream():
            process(frame)
    """
    
    def __init__(
        self,
        source: Union[int, str] = 0,
        target_size: tuple = (1280, 720), # Dashboard aspect ratio matches 1280x720
        fps_limit: Optional[int] = 30,
        buffer_size: int = 3,
        retry_interval: float = 5.0
    ):
        self.source = source
        self.target_size = target_size  # (width, height) for model/hud input
        self.fps_limit = fps_limit
        self.buffer_size = buffer_size
        self.retry_interval = retry_interval
        
        self.cap = None
        self.is_streaming = False
        self.frame_time = 1.0 / fps_limit if fps_limit else 0
        
        # Threading for non-blocking capture
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.capture_thread = None
        self._stop_event = threading.Event()
        
        # Stats
        self.frames_captured = 0
        self.frames_dropped = 0
        self.start_time = None
        
        # Synthetic shelf interactive demo variables
        self.items = []
        self.last_hand_update = time.time()
        self.shopper_active = False
        self.hand_pos = [0, 0]
        self.hand_target = [0, 0]
        self.hand_speed = 15
        self.target_item = None
        self.action_state = "idle"
        
        if self.source == "synthetic":
            self.initialize_synthetic_shelf()

    def initialize_synthetic_shelf(self):
        """Pre-define item positions on a 3-shelf grid for synthetic mode"""
        self.shelves_y = [200, 400, 600]
        self.shelf_height = 80
        
        labels = [
            ("Soda", (0, 0, 255)),     # Red
            ("Chips", (0, 200, 255)),  # Yellow
            ("Water", (255, 100, 0))   # Blue
        ]
        
        item_id = 0
        for shelf_idx, y_center in enumerate(self.shelves_y):
            label, color = labels[shelf_idx]
            for i in range(6):
                x_center = 180 + i * 180
                w, h = (60, 100) if label == "Soda" else ((80, 80) if label == "Chips" else (50, 110))
                self.items.append({
                    "id": item_id,
                    "label": label,
                    "color": color,
                    "x": x_center,
                    "y": y_center - h // 2,
                    "w": w,
                    "h": h,
                    "status": 1, # 1: normal, 0: OOS, 2: misplaced
                    "last_touched": 0
                })
                item_id += 1

    def _connect(self) -> bool:
        """Establish connection to video source."""
        if self.source == "synthetic":
            logger.info("Connected: Synthetic Interactive Demo Feed active")
            return True
            
        try:
            if self.cap is not None:
                self.cap.release()
                
            source = int(self.source) if str(self.source).isdigit() else self.source
            
            self.cap = cv2.VideoCapture(source)
            
            # Optimize OpenCV backend
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_limit or 30)
            
            # RTSP optimizations
            if isinstance(source, str) and source.startswith('rtsp'):
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                self.cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.CAP_RTSP_TRANSPORT_TCP)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open source: {self.source}")
                return False
                
            # Log source info
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Connected: {width}x{height} @ {fps:.1f}fps")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def _capture_loop(self):
        """Background thread: continuously read/generate frames."""
        while not self._stop_event.is_set():
            if self.source != "synthetic" and (self.cap is None or not self.cap.isOpened()):
                logger.info("Reconnecting...")
                if not self._connect():
                    time.sleep(self.retry_interval)
                    continue
            
            if self.source == "synthetic":
                frame = self._generate_synthetic_frame()
                ret = True
                # Cap synthetic rendering to avoid CPU spikes
                time.sleep(0.033) # ~30 FPS
            else:
                ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("Frame read failed")
                if isinstance(self.source, str) and Path(self.source).is_file():
                    # Video file ended
                    self._stop_event.set()
                    break
                time.sleep(0.1)
                continue
            
            # Resize and preprocess (letterbox padding to target size)
            processed = self._preprocess(frame)
            
            # Drop old frames if buffer full (keep latest)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put_nowait({
                    'original': frame,
                    'processed': processed,
                    'timestamp': time.time(),
                    'frame_num': self.frames_captured
                })
                self.frames_captured += 1
            except queue.Full:
                pass
    
    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generates realistic synthetic shelf frames with simulated shopper interactions"""
        w, h = 1280, 720
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = [210, 220, 230] # Beige wall base
        
        # Draw wooden shelves
        for y in self.shelves_y:
            cv2.rectangle(frame, (50, y), (w - 50, y + 20), (42, 42, 165), -1) # Dark wood
            cv2.rectangle(frame, (50, y), (w - 50, y + 20), (80, 100, 200), 2)
            cv2.rectangle(frame, (50, y + 16), (w - 50, y + 20), (0, 215, 255), -1) # Yellow line tag
            
        # Shopper simulator state updates
        now = time.time()
        if not self.shopper_active and now - self.last_hand_update > random.uniform(8.0, 15.0):
            self.shopper_active = True
            self.hand_pos = [random.choice([0, w]), h]
            
            available_items = [i for i in self.items if i["status"] != 0]
            oos_items = [i for i in self.items if i["status"] == 0]
            
            if oos_items and random.random() < 0.4:
                self.target_item = random.choice(oos_items)
                self.action_state = "restocking"
            elif available_items:
                self.target_item = random.choice(available_items)
                self.action_state = "reaching"
            else:
                self.shopper_active = False
                
            if self.target_item:
                self.hand_target = [self.target_item["x"], self.target_item["y"] + self.target_item["h"] // 2]
            self.last_hand_update = now

        if self.shopper_active and self.target_item:
            dx = self.hand_target[0] - self.hand_pos[0]
            dy = self.hand_target[1] - self.hand_pos[1]
            dist = np.hypot(dx, dy)
            
            if dist > 15:
                self.hand_pos[0] += int(dx / dist * self.hand_speed)
                self.hand_pos[1] += int(dy / dist * self.hand_speed)
            else:
                if self.action_state == "reaching":
                    self.target_item["status"] = 0
                    self.target_item["last_touched"] = now
                    self.action_state = "retreating"
                    self.hand_target = [random.choice([0, w]), h]
                elif self.action_state == "restocking":
                    self.target_item["status"] = 1
                    self.target_item["last_touched"] = now
                    if random.random() < 0.25:
                        original_label = self.target_item["label"]
                        other_labels = [l for l in ["Soda", "Chips", "Water"] if l != original_label]
                        self.target_item["label"] = random.choice(other_labels)
                        self.target_item["status"] = 2 
                    self.action_state = "retreating"
                    self.hand_target = [random.choice([0, w]), h]
                elif self.action_state == "retreating":
                    self.shopper_active = False
                    self.target_item = None
                    self.action_state = "idle"
                    
        # Minor ambient random grabs
        if not self.shopper_active and random.random() < 0.005:
            itm = random.choice(self.items)
            itm["status"] = 0 if itm["status"] != 0 else 1
            itm["last_touched"] = now

        # Draw items
        for item in self.items:
            if item["status"] == 0:
                continue
            x, y, iw, ih = item["x"], item["y"], item["w"], item["h"]
            draw_color = item["color"]
            if item["status"] == 2:
                if item["label"] == "Soda": draw_color = (0, 0, 255)
                elif item["label"] == "Chips": draw_color = (0, 200, 255)
                else: draw_color = (255, 100, 0)
                
            if now - item["last_touched"] < 2.0:
                cv2.rectangle(frame, (x - 20 - int(np.sin(now * 10) * 5), y - 10), (x + iw + 20 + int(np.sin(now * 10) * 5), y + ih + 10), (255, 255, 255), 2)
            
            if item["label"] == "Soda":
                cv2.rectangle(frame, (x + iw // 3, y), (x + 2 * iw // 3, y + ih // 4), draw_color, -1)
                cv2.rectangle(frame, (x, y + ih // 4), (x + iw, y + ih), draw_color, -1)
                cv2.rectangle(frame, (x + iw // 3 - 2, y - 10), (x + 2 * iw // 3 + 2, y), (0, 255, 255), -1)
                cv2.rectangle(frame, (x, y + ih // 2), (x + iw, y + 2 * ih // 3), (255, 255, 255), -1)
                cv2.putText(frame, "SODA", (x + 5, y + 3 * ih // 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            elif item["label"] == "Chips":
                pts = np.array([[x + iw // 2, y], [x + iw, y + 20], [x + iw - 10, y + ih - 20], [x + iw // 2, y + ih], [x + 10, y + ih - 20], [x, y + 20]], np.int32)
                cv2.fillPoly(frame, [pts], draw_color)
                cv2.line(frame, (x + 5, y + 5), (x + iw - 5, y + 5), (200, 200, 200), 2)
                cv2.line(frame, (x + 15, y + ih - 5), (x + iw - 15, y + ih - 5), (200, 200, 200), 2)
                cv2.rectangle(frame, (x + 15, y + ih // 3), (x + iw - 15, y + 2 * ih // 3), (255, 255, 255), -1)
                cv2.putText(frame, "CHIPS", (x + 20, y + ih // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            else:
                cv2.circle(frame, (x + iw // 2, y), 8, (0, 165, 255), -1)
                cv2.rectangle(frame, (x, y + 10), (x + iw, y + ih), draw_color, -1)
                for r_y in range(y + 25, y + ih - 15, 20):
                    cv2.line(frame, (x + 5, r_y), (x + iw - 5, r_y), (255, 255, 255), 1)
                cv2.rectangle(frame, (x, y + ih // 3 + 10), (x + iw, y + ih // 3 + 30), (255, 255, 255), -1)
                cv2.putText(frame, "H2O", (x + 8, y + ih // 3 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Draw arm
        if self.shopper_active:
            cv2.line(frame, (self.hand_target[0], self.hand_target[1]), (self.hand_pos[0], self.hand_pos[1]), (80, 80, 80), 1)
            cv2.line(frame, (self.hand_pos[0], self.hand_pos[1]), (self.hand_pos[0] + (1 if self.hand_pos[0] < w // 2 else -1) * 300, h + 200), (45, 105, 154), 22)
            cv2.line(frame, (self.hand_pos[0], self.hand_pos[1]), (self.hand_pos[0] + (1 if self.hand_pos[0] < w // 2 else -1) * 300, h + 200), (80, 160, 220), 12)
            cv2.circle(frame, (self.hand_pos[0], self.hand_pos[1]), 25, (140, 190, 240), -1)
            cv2.circle(frame, (self.hand_pos[0], self.hand_pos[1]), 8, (0, 255, 0) if self.action_state == "retreating" else (0, 0, 255), -1)

        # Ambient frame noise
        noise = np.random.normal(0, 1.2, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        cv2.rectangle(frame, (20, 20), (250, 60), (0, 0, 0), -1)
        cv2.putText(frame, "CAM-01 [LIVE]", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(frame, (230, 40), 6, (0, 0, 255) if int(now) % 2 == 0 else (0, 50, 0), -1)
        
        return frame

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to model input size while maintaining aspect ratio (letterboxing)."""
        h, w = frame.shape[:2]
        tw, th = self.target_size
        
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        padded = np.full((th, tw, 3), 114, dtype=np.uint8)  # Gray padding
        y_offset = (th - new_h) // 2
        x_offset = (tw - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    
    def start(self):
        """Start capture thread."""
        self._stop_event.clear()
        self.start_time = time.time()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.is_streaming = True
        logger.info("Capture started")
        return self
    
    def read(self, timeout: float = 1.0) -> Optional[dict]:
        """Get latest frame (non-blocking with timeout)."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stream(self) -> Iterator[dict]:
        """Generator for streaming frames."""
        self.start()
        try:
            while self.is_streaming and not self._stop_event.is_set():
                frame_data = self.read(timeout=2.0)
                if frame_data is None:
                    continue
                
                # FPS limiting
                if self.frame_time > 0:
                    elapsed = time.time() - self.start_time
                    expected_frames = elapsed / self.frame_time
                    if self.frames_captured > expected_frames:
                        time.sleep(self.frame_time - (elapsed % self.frame_time))
                
                yield frame_data
        finally:
            self.stop()
    
    def get_stats(self) -> dict:
        """Get capture statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'fps_actual': self.frames_captured / elapsed if elapsed > 0 else 0,
            'queue_size': self.frame_queue.qsize(),
            'is_alive': self.capture_thread.is_alive() if self.capture_thread else False
        }
    
    def stop(self):
        """Stop capture and release resources."""
        self._stop_event.set()
        self.is_streaming = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("Capture stopped")

# Standalone test
if __name__ == "__main__":
    import sys
    
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    print(f"Testing capture from: {source}")
    
    cap = FrameCapture(source=source)
    
    try:
        for frame_data in cap.stream():
            frame = frame_data['original']
            cv2.imshow("Capture Test (press Q to quit)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            if frame_data['frame_num'] % 30 == 0:
                stats = cap.get_stats()
                print(f"Stats: {stats}")
    finally:
        cap.stop()
        cv2.destroyAllWindows()
