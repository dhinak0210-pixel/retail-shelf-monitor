"""
detector.py - YOLOv8 object detection with retail-specific optimizations
"""

import cv2
import numpy as np
import torch
import time
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[Detector] 'ultralytics' package not found. Running in high-fidelity SIMULATION mode.")

logger = logging.getLogger("detector")

@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[int, int]
    area: int
    
    def to_dict(self) -> dict:
        return {
            'bbox': self.bbox,
            'confidence': round(self.confidence, 3),
            'class_id': self.class_id,
            'class_name': self.class_name,
            'center': self.center,
            'area': self.area
        }


class ShelfDetector:
    """
    YOLOv8-based shelf product detector.
    
    Features:
    - Automatic device selection (CUDA > MPS > CPU)
    - Batch inference support
    - Confidence filtering
    - NMS optimization
    - Custom class mapping for retail SKUs
    """
    
    # Default COCO classes relevant to retail (or use custom trained model)
    DEFAULT_CLASSES = {
        0: 'person', 39: 'bottle', 40: 'wine glass', 41: 'cup',
        42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
        54: 'donut', 55: 'cake', 73: 'book', 74: 'clock',
        75: 'vase', 76: 'scissors', 77: 'teddy bear'
    }
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        classes: Optional[Dict[int, str]] = None,
        half: bool = False,  # FP16 inference
        force_simulation: bool = False
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes or self.DEFAULT_CLASSES
        self.simulation_mode = force_simulation or not ULTRALYTICS_AVAILABLE
        
        self.inference_times = []
        self.total_inferences = 0
        self.model = None

        if not self.simulation_mode:
            try:
                # Auto-select device
                self.device = device or self._auto_select_device()
                self.half = half and self.device != 'cpu'
                
                logger.info(f"Loading model: {model_path} on {self.device}")
                self.model = YOLO(model_path)
                self.model.to(self.device)
                
                # Warmup
                self._warmup()
            except Exception as e:
                logger.error(f"Failed to load YOLOv8 model: {e}. Falling back to SIMULATION mode.")
                self.simulation_mode = True
        
    def _auto_select_device(self) -> str:
        """Select best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def _warmup(self, runs: int = 3):
        """Warmup inference to initialize GPU kernels."""
        dummy = torch.zeros(1, 3, 640, 640, device=self.device)
        logger.info("Warming up model...")
        for _ in range(runs):
            self.model.predict(dummy, verbose=False)
        logger.info("Warmup complete")
    
    def predict(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single frame.
        
        Args:
            frame: BGR image (H, W, 3) or RGB preprocessed
            
        Returns:
            List of Detection objects
        """
        if self.simulation_mode:
            return []
            
        start_time = time.perf_counter()
        
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            half=self.half,
            verbose=False,
            stream=False
        )[0]
        
        # Parse results
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = (x2 - x1) * (y2 - y1)
                
                # Map class name
                class_name = self.classes.get(cls_id, f"class_{cls_id}")
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=class_name,
                    center=center,
                    area=area
                ))
        
        # Track performance
        inference_time = time.perf_counter() - start_time
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        self.total_inferences += 1
        
        return detections
    
    def predict_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Batch inference for multiple frames."""
        if self.simulation_mode:
            return [[] for _ in frames]
            
        results = self.model.predict(
            frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            half=self.half,
            verbose=False,
            stream=False
        )
        
        all_detections = []
        for result in results:
            detections = []
            if result.boxes is not None:
                for box, conf, cls_id in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int)
                ):
                    x1, y1, x2, y2 = map(int, box)
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=self.classes.get(int(cls_id), f"class_{int(cls_id)}"),
                        center=((x1 + x2) // 2, (y1 + y2) // 2),
                        area=(x2 - x1) * (y2 - y1)
                    ))
            all_detections.append(detections)
        
        return all_detections
    
    def detect(self, frame: np.ndarray, capture_instance=None) -> List[dict]:
        """
        Runs object detection on the provided frame.
        Maps YOLO outputs or falls back to simulation mode coordinate tracking.
        """
        if self.simulation_mode or (capture_instance and capture_instance.source == "synthetic"):
            return self._run_simulation_detection(capture_instance)
            
        try:
            detections = []
            raw_dets = self.predict(frame)
            for d in raw_dets:
                class_name = d.class_name
                class_id = d.class_id
                
                # Custom mapping for retail SKUs (Bottle -> Soda/Water, Cup -> Chips)
                mapped_name = class_name
                if class_name in ["bottle", "wine glass"]:
                    mapped_name = "Soda" if class_id == 39 else "Water"
                elif class_name in ["cup", "bowl"]:
                    mapped_name = "Chips"
                    
                detections.append({
                    "box": list(d.bbox),
                    "confidence": round(d.confidence, 2),
                    "class": mapped_name,
                    "class_id": class_id
                })
            return detections
        except Exception as e:
            logger.error(f"Inference error: {e}. Falling back to simulation mode.")
            return self._run_simulation_detection(capture_instance)

    def _run_simulation_detection(self, capture_instance) -> List[dict]:
        """Retrieves exact coordinates of active simulated products to emulate YOLO detections"""
        detections = []
        if capture_instance and hasattr(capture_instance, 'items'):
            for item in capture_instance.items:
                if item["status"] != 0: # In stock or misplaced
                    x, y, w, h = item["x"], item["y"], item["w"], item["h"]
                    
                    # Apply slight random box jitter to mimic YOLO fluctuations
                    jx = random.randint(-2, 2)
                    jy = random.randint(-2, 2)
                    jw = random.randint(-1, 1)
                    jh = random.randint(-1, 1)
                    
                    cls_name = item["label"]
                    cls_id = 0 if cls_name == "Soda" else (1 if cls_name == "Chips" else 2)
                    
                    detections.append({
                        "box": [
                            max(0, x + jx),
                            max(0, y + jy),
                            min(1280, x + w + jx + jw),
                            min(720, y + h + jy + jh)
                        ],
                        "confidence": round(random.uniform(0.85, 0.98), 2),
                        "class": cls_name,
                        "class_id": cls_id
                    })
                    
            # Add simulated shopper person class
            if getattr(capture_instance, 'shopper_active', False):
                hx = capture_instance.hand_pos[0]
                x1 = max(0, hx - 120)
                y1 = 350
                x2 = min(1280, hx + 120)
                y2 = 720
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": round(random.uniform(0.93, 0.98), 2),
                    "class": "person",
                    "class_id": 0 # COCO person class ID
                })
        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        output = frame.copy()
        
        # Default color palette
        if color_map is None:
            np.random.seed(42)
            unique_classes = list(set(d.class_name for d in detections))
            colors = np.random.randint(0, 255, size=(len(unique_classes), 3), dtype=np.uint8)
            color_map = {cls: tuple(map(int, colors[i])) for i, cls in enumerate(unique_classes)}
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = color_map.get(det.class_name, (0, 255, 0))
            
            # Box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Label background
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            
            # Label text
            cv2.putText(output, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Center dot
            cv2.circle(output, det.center, 3, (0, 0, 255), -1)
        
        return output
    
    def get_stats(self) -> dict:
        """Get inference statistics."""
        avg_time = np.mean(self.inference_times) if self.inference_times else 0
        device_name = getattr(self, 'device', 'simulation')
        model_name = getattr(self.model, 'model_name', 'sim-model') if self.model else 'simulation'
        return {
            'device': device_name,
            'total_inferences': self.total_inferences,
            'avg_inference_ms': round(avg_time * 1000, 2),
            'fps_capacity': round(1 / avg_time, 1) if avg_time > 0 else 0,
            'model': str(model_name)
        }


# Test
if __name__ == "__main__":
    import sys
    
    print("Testing ShelfDetector...")
    detector = ShelfDetector(conf_threshold=0.3)
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = detector.predict(frame)
            annotated = detector.draw_detections(frame, detections)
            
            # Stats overlay
            stats = detector.get_stats()
            cv2.putText(annotated, f"FPS cap: {stats['fps_capacity']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Detector Test (Q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Final stats: {detector.get_stats()}")
