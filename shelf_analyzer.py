"""
shelf_analyzer.py - Retail shelf planogram analysis
Maps detections to grid cells, identifies empty slots and misplacements
"""

import numpy as np
import cv2
import time
import json
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from detector import Detection

@dataclass
class GridCell:
    """Single cell in the planogram grid."""
    row: int
    col: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords
    expected_class: Optional[str] = None
    current_detection: Optional[Detection] = None
    is_occupied: bool = False
    oos_start_time: Optional[float] = None  # When OOS started
    oos_duration: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'row': self.row,
            'col': self.col,
            'expected': self.expected_class,
            'occupied': self.is_occupied,
            'current_class': self.current_detection.class_name if self.current_detection else None,
            'confidence': round(self.current_detection.confidence, 3) if self.current_detection else 0,
            'oos_duration': round(self.oos_duration, 1)
        }


class PlanogramGrid:
    """
    Divides shelf image into R×C grid for planogram compliance.
    
    Features:
    - Automatic grid sizing based on shelf dimensions
    - Cell occupancy tracking
    - OOS (Out-of-Stock) detection with duration
    - Misplacement detection
    - Heatmap generation for OOS frequency
    """
    
    def __init__(
        self,
        rows: int = 4,
        cols: int = 6,
        frame_size: Tuple[int, int] = (640, 640),
        oos_threshold_seconds: float = 3.0,
        history_size: int = 30  # Frames to keep for heatmap
    ):
        self.rows = rows
        self.cols = cols
        self.frame_width, self.frame_height = frame_size
        self.oos_threshold = oos_threshold_seconds
        self.history_size = history_size
        
        # ROI coordinate bounds (default to entire frame)
        self.roi_x_min = 0
        self.roi_y_min = 0
        self.roi_x_max = self.frame_width
        self.roi_y_max = self.frame_height
        
        # Calculate cell dimensions
        self.cell_width = self.frame_width // cols
        self.cell_height = self.frame_height // rows
        
        # Initialize grid
        self.grid: List[List[GridCell]] = []
        self._init_grid()
        
        # OOS tracking history for heatmap
        self.oos_history: deque = deque(maxlen=history_size)
        
        # Predictive stockout forecasting engines
        self.stock_history = defaultdict(list)
        self.velocity_rates = defaultdict(float)
        self.forecast_predictions = {}
        
        # Shopper tracking and foot-traffic dwell metrics
        self.shopper_tracks = []
        self.dwell_times = defaultdict(float)
        self.last_track_update = time.time()
        
        # Camera health monitoring fields
        self.is_tampered = False
        self.tamper_reason = ""
        self.blur_val = 0.0
        self.brightness_val = 0.0
        
        # Alert tracking (to avoid spam)
        self.active_alerts: Set[Tuple[int, int]] = set()
        self.alert_cooldown: Dict[Tuple[int, int], float] = {}
        
        # Stats
        self.total_oos_events = 0
        self.total_misplacements = 0
        self.frame_count = 0
        
    def _init_grid(self):
        """Initialize grid cells with pixel coordinates."""
        self.grid = []
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                x1 = self.roi_x_min + c * self.cell_width
                y1 = self.roi_y_min + r * self.cell_height
                x2 = x1 + self.cell_width
                y2 = y1 + self.cell_height
                row_cells.append(GridCell(
                    row=r, col=c,
                    bbox=(x1, y1, x2, y2)
                ))
            self.grid.append(row_cells)

    def set_roi(self, x_min: int, y_min: int, x_max: int, y_max: int):
        """Redefines the entire grid boundary dynamically based on visual draw interactions."""
        self.roi_x_min = max(0, x_min)
        self.roi_y_min = max(0, y_min)
        self.roi_x_max = min(self.frame_width, x_max)
        self.roi_y_max = min(self.frame_height, y_max)
        
        width = self.roi_x_max - self.roi_x_min
        height = self.roi_y_max - self.roi_y_min
        self.cell_width = width // self.cols
        self.cell_height = height // self.rows
        
        # Update coordinates on existing cell instances to preserve planogram SKU mappings!
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = self.roi_x_min + c * self.cell_width
                y1 = self.roi_y_min + r * self.cell_height
                x2 = x1 + self.cell_width
                y2 = y1 + self.cell_height
                self.grid[r][c].bbox = (x1, y1, x2, y2)
    
    def set_planogram(self, planogram: Dict[Tuple[int, int], str]):
        """
        Set expected product layout.
        
        Args:
            planogram: Dict mapping (row, col) -> expected class name
        """
        for (r, c), expected_class in planogram.items():
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r][c].expected_class = expected_class
    
    def load_planogram_from_json(self, path: str):
        """Load planogram from JSON file."""
        with open(path) as f:
            data = json.load(f)
            planogram = {
                (item['row'], item['col']): item['expected_class']
                for item in data['layout']
            }
            self.set_planogram(planogram)
    
    def map_detections(self, detections: List[Detection]) -> List[GridCell]:
        """
        Map detections to grid cells based on center points.
        
        Returns:
            Updated grid cells
        """
        current_time = time.time()
        self.frame_count += 1
        
        # Track shopper traffic paths and dwell times
        people = [d for d in detections if d.class_name == "person"]
        dt = current_time - getattr(self, 'last_track_update', current_time)
        self.last_track_update = current_time
        
        if people:
            p = people[0]
            cx, cy = p.center
            self.shopper_tracks.append({"x": cx, "y": cy, "time": current_time})
            
            # Map center to column index
            col_width = self.frame_width // self.cols
            col_idx = int(cx // col_width)
            col_idx = max(0, min(col_idx, self.cols - 1))
            
            if dt < 1.0:
                self.dwell_times[col_idx] += dt
                
        # Fade out old tracks (older than 10 seconds for visual responsiveness)
        self.shopper_tracks = [t for t in self.shopper_tracks if current_time - t["time"] <= 10.0]
        
        # Filter out human detections so we only map product SKUs onto the grid
        product_detections = [d for d in detections if d.class_name != "person"]
        
        # Reset current state
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                cell.current_detection = None
                cell.is_occupied = False
        
        # Map detections to cells
        for det in product_detections:
            cx, cy = det.center
            if self.roi_x_min <= cx <= self.roi_x_max and self.roi_y_min <= cy <= self.roi_y_max:
                col = int((cx - self.roi_x_min) // self.cell_width)
                row = int((cy - self.roi_y_min) // self.cell_height)
                
                # Dynamic boundaries constraints check
                col = min(col, self.cols - 1)
                row = min(row, self.rows - 1)
                
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    cell = self.grid[row][col]
                    # Keep highest confidence detection per cell
                    if cell.current_detection is None or det.confidence > cell.current_detection.confidence:
                        cell.current_detection = det
                        cell.is_occupied = True
        
        # Update OOS tracking
        oos_snapshot = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                
                if not cell.is_occupied and cell.expected_class is not None:
                    # Cell should have product but is empty
                    if cell.oos_start_time is None:
                        cell.oos_start_time = current_time
                    cell.oos_duration = current_time - cell.oos_start_time
                    
                    # Mark for heatmap
                    if cell.oos_duration > 0:
                        oos_snapshot[r, c] = min(cell.oos_duration / 10.0, 1.0)  # Normalize
                    
                    # Check for alert threshold
                    if cell.oos_duration >= self.oos_threshold:
                        self.total_oos_events += 1
                else:
                    # Product present or no expectation
                    cell.oos_start_time = None
                    cell.oos_duration = 0.0
        
        self.oos_history.append(oos_snapshot)
        
        # Run real-time predictive stockout forecasting
        self.compute_forecasting()
        
        return [cell for row in self.grid for cell in row]

    def compute_forecasting(self):
        """Calculates rolling consumption velocities and estimates time to empty."""
        now = time.time()
        
        # Calculate current counts of each SKU class present
        counts = defaultdict(int)
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                if cell.is_occupied and cell.current_detection:
                    counts[cell.current_detection.class_name] += 1
                    
        # Seed all configured SKU names
        classes_to_track = {"Soda", "Chips", "Water"}
        for cls in classes_to_track:
            self.stock_history[cls].append((now, counts[cls]))
            
            # Prune records older than 30 seconds to capture rapid/responsive retail grab velocities
            self.stock_history[cls] = [item for item in self.stock_history[cls] if now - item[0] <= 30.0]
            
            history = self.stock_history[cls]
            if len(history) >= 2:
                t0, c0 = history[0]
                tn, cn = history[-1]
                elapsed = tn - t0
                
                if elapsed > 2.0:
                    # Stock drop represent consumption grabbing
                    stock_decrease = max(0, c0 - cn)
                    raw_rate = stock_decrease / elapsed
                    
                    # Apply low-pass exponential moving average to filter noise
                    self.velocity_rates[cls] = 0.8 * self.velocity_rates[cls] + 0.2 * raw_rate
                    
            # Compute Time to Empty (TTE) projections
            current_stock = counts[cls]
            rate = self.velocity_rates[cls]
            
            if rate > 0.005 and current_stock > 0:
                tte = current_stock / rate
                risk = "critical" if tte < 15.0 else ("warning" if tte < 35.0 else "low")
                self.forecast_predictions[cls] = {
                    "sku": cls,
                    "stock": current_stock,
                    "rate_per_min": round(rate * 60.0, 1),
                    "tte_seconds": round(tte, 1),
                    "risk": risk,
                    "message": f"Critical stockout in {round(tte, 1)}s ({round(rate * 60.0, 1)} grabs/min)"
                }
            elif current_stock == 0:
                self.forecast_predictions[cls] = {
                    "sku": cls,
                    "stock": 0,
                    "rate_per_min": round(rate * 60.0, 1),
                    "tte_seconds": 0.0,
                    "risk": "empty",
                    "message": "OUT OF STOCK"
                }
            else:
                self.forecast_predictions[cls] = {
                    "sku": cls,
                    "stock": current_stock,
                    "rate_per_min": 0.0,
                    "tte_seconds": None,
                    "risk": "stable",
                    "message": "Stock level stable"
                }
    
    def audit_camera_health(self, frame: np.ndarray):
        """Audits video stream frame for physical tampering, blur, and lens occlusion."""
        if frame is None:
            return
            
        try:
            import cv2
            # Check if frame is grayscale or color, convert to gray for mathematical auditing
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
                
            # 1. Measure sharpness using Laplacian variance
            self.blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Measure overall illumination level
            self.brightness_val = float(gray.mean())
            
            # 3. Assess tampering states
            # Standard threshold: Laplacian variance < 30 (extreme blur), brightness < 20 (blocked)
            if self.brightness_val < 20.0:
                self.is_tampered = True
                self.tamper_reason = "Lens Occluded (Lens completely covered, dark, or blocked)"
            elif self.blur_val < 30.0:
                self.is_tampered = True
                self.tamper_reason = "Out of Focus (Lens smudged, blurry, or misaligned)"
            else:
                self.is_tampered = False
                self.tamper_reason = ""
                
        except Exception as e:
            self.is_tampered = False
            self.tamper_reason = f"Health check failed: {e}"
            
    def check_misplacements(self) -> List[Dict]:
        """Detect products in wrong locations."""
        misplacements = []
        
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                if cell.is_occupied and cell.expected_class:
                    actual = cell.current_detection.class_name
                    if actual != cell.expected_class:
                        misplacements.append({
                            'row': r, 'col': c,
                            'expected': cell.expected_class,
                            'actual': actual,
                            'confidence': cell.current_detection.confidence
                        })
                        self.total_misplacements += 1
        
        return misplacements
    
    def get_heatmap(self) -> np.ndarray:
        """
        Generate OOS frequency heatmap.
        
        Returns:
            (rows, cols) float array normalized 0-1
        """
        if not self.oos_history:
            return np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # Average over history
        heatmap = np.mean(list(self.oos_history), axis=0)
        
        # Normalize
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
        
        return heatmap
    
    def get_stats(self) -> dict:
        """Get current shelf statistics."""
        total_cells = self.rows * self.cols
        occupied = sum(1 for r in self.grid for c in r if c.is_occupied)
        expected_cells = sum(1 for r in self.grid for c in r if c.expected_class)
        oos_now = sum(1 for r in self.grid for c in r 
                     if c.expected_class and not c.is_occupied)
        
        fill_rate = occupied / total_cells if total_cells > 0 else 0
        planogram_compliance = ((expected_cells - oos_now) / expected_cells 
                               if expected_cells > 0 else 0)
        
        return {
            'fill_rate': round(fill_rate * 100, 1),
            'oos_count': oos_now,
            'oos_total_events': self.total_oos_events,
            'misplacements': len(self.check_misplacements()),
            'total_misplacements': self.total_misplacements,
            'planogram_compliance': round(planogram_compliance * 100, 1),
            'total_cells': total_cells,
            'occupied': occupied,
            'expected_cells': expected_cells,
            'shopper_tracks': self.shopper_tracks,
            'dwell_times': {str(k): round(v, 1) for k, v in self.dwell_times.items()}
        }
    
    def draw_grid(self, frame: np.ndarray, show_labels: bool = True) -> np.ndarray:
        """Draw grid overlay on frame."""
        output = frame.copy()
        heatmap = self.get_heatmap()
        
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                x1, y1, x2, y2 = cell.bbox
                
                # Cell background based on state
                if cell.is_occupied:
                    if cell.expected_class and cell.current_detection.class_name != cell.expected_class:
                        color = (0, 165, 255)  # Orange: misplacement
                        alpha = 0.3
                    else:
                        color = (0, 255, 0)  # Green: correct
                        alpha = 0.2
                else:
                    if cell.expected_class:
                        # OOS - intensity based on duration
                        intensity = min(cell.oos_duration / self.oos_threshold, 1.0)
                        red = int(255 * intensity)
                        color = (0, 0, red)
                        alpha = 0.2 + (0.3 * intensity)
                    else:
                        color = (128, 128, 128)  # Gray: no expectation
                        alpha = 0.1
                
                # Draw filled rectangle
                overlay = output.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                output = cv2.addWeighted(output, 1 - alpha, overlay, alpha, 0)
                
                # Grid lines
                cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 1)
                
                # Labels
                if show_labels:
                    label = ""
                    if cell.expected_class:
                        label = f"E:{cell.expected_class[:8]}"
                    if cell.is_occupied:
                        label += f"\nA:{cell.current_detection.class_name[:8]}"
                    elif cell.expected_class:
                        label += f"\nOOS:{cell.oos_duration:.1f}s"
                    
                    y_offset = y1 + 20
                    for i, line in enumerate(label.split('\n')):
                        if line:
                            cv2.putText(output, line, (x1 + 5, y_offset + i * 15),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Draw heatmap overlay in corner
        if heatmap.max() > 0:
            hm_size = 100
            hm_resized = cv2.resize((heatmap * 255).astype(np.uint8), 
                                   (hm_size, hm_size), interpolation=cv2.INTER_NEAREST)
            hm_colored = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
            
            # Place in top-right corner
            h, w = output.shape[:2]
            output[10:10+hm_size, w-hm_size-10:w-10] = hm_colored
            
            # Label
            cv2.putText(output, "OOS Heatmap", (w-hm_size-10, 10+hm_size+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return output


class ShelfAnalyzer:
    """
    Backward-compatible wrapper integrating advanced PlanogramGrid 
    capabilities into server.py workflows.
    """
    def __init__(self):
        # 3 shelves (rows) by 6 columns, mapped onto HD 1280x720 video dimensions
        self.grid_engine = PlanogramGrid(rows=3, cols=6, frame_size=(1280, 720), oos_threshold_seconds=3.0)
        
        # Seed expected layout default planogram classes
        self.initialize_default_planogram()
        
        self.active_alerts = {}
        self.alert_cooldown = 10.0 # seconds

    def initialize_default_planogram(self):
        new_planogram = {}
        for c in range(6):
            new_planogram[(0, c)] = "Soda"
            new_planogram[(1, c)] = "Chips"
            new_planogram[(2, c)] = "Water"
        self.grid_engine.set_planogram(new_planogram)

    @property
    def planogram(self) -> dict:
        """
        Dynamically formats expected layouts for stats.js planogram configurator updates.
        """
        shelves_list = []
        for r in range(3):
            shelf_id = r + 1
            shelf_name = "Top Shelf - Soda" if r == 0 else ("Middle Shelf - Chips" if r == 1 else "Bottom Shelf - Water")
            y_start = r * 240
            y_end = (r + 1) * 240
            
            slots_list = []
            for c in range(6):
                cell = self.grid_engine.grid[r][c]
                status = "OK" if cell.is_occupied and cell.current_detection.class_name == cell.expected_class else (
                    "MISPLACED" if cell.is_occupied else "OOS"
                )
                slots_list.append({
                    "slot_id": c + 1,
                    "x_range": [c * 213, (c + 1) * 213],
                    "status": status,
                    "current_item": cell.current_detection.class_name if cell.is_occupied else None,
                    "expected": cell.expected_class
                })
                
            shelves_list.append({
                "id": shelf_id,
                "name": shelf_name,
                "y_range": [y_start, y_end],
                "expected": self.grid_engine.grid[r][0].expected_class or "Soda",
                "color": (0, 0, 255) if r == 0 else ((0, 200, 255) if r == 1 else (255, 100, 0)),
                "slots": slots_list
            })
            
        return {"shelves": shelves_list}

    def clear_alerts(self):
        """Resets active alert states and OOS counters"""
        self.active_alerts.clear()
        self.grid_engine.total_oos_events = 0
        self.grid_engine.total_misplacements = 0
        for r in range(self.grid_engine.rows):
            for c in range(self.grid_engine.cols):
                cell = self.grid_engine.grid[r][c]
                cell.oos_start_time = None
                cell.oos_duration = 0.0

    def update_planogram_config(self, config_data) -> bool:
        """Dynamic planogram configuration updates via REST API"""
        if "shelves" in config_data:
            new_plan = {}
            for s_cfg in config_data["shelves"]:
                row_idx = s_cfg.get("id", 1) - 1
                expected = s_cfg.get("expected", "Soda")
                if 0 <= row_idx < 3:
                    for col_idx in range(6):
                        new_plan[(row_idx, col_idx)] = expected
            self.grid_engine.set_planogram(new_plan)
            return True
        return False

    def analyze(self, detections: List[dict]) -> dict:
        """
        Maps dictionary-based detections into Detection dataclasses, 
        evaluates compliance, and structures UI telemetry models.
        """
        # Convert raw detections into Dataclass Detection forms
        dataclass_dets = []
        for det in detections:
            box = det["box"]
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            area = (box[2] - box[0]) * (box[3] - box[1])
            dataclass_dets.append(Detection(
                bbox=tuple(box),
                confidence=det["confidence"],
                class_id=det["class_id"],
                class_name=det["class"],
                center=(cx, cy),
                area=area
            ))

        # Run core grid mapping logic
        self.grid_engine.map_detections(dataclass_dets)
        
        # Compile stats and alerts
        stats = self.grid_engine.get_stats()
        alerts = []
        now = time.time()

        for r in range(self.grid_engine.rows):
            for c in range(self.grid_engine.cols):
                cell = self.grid_engine.grid[r][c]
                shelf_id = r + 1
                slot_id = c + 1
                shelf_name = "Top Shelf - Soda" if r == 0 else ("Middle Shelf - Chips" if r == 1 else "Bottom Shelf - Water")
                
                # Check for OOS state alerts
                if not cell.is_occupied and cell.expected_class:
                    alert_key = f"oos_s{shelf_id}_p{slot_id}"
                    if alert_key not in self.active_alerts or (now - self.active_alerts[alert_key] > self.alert_cooldown):
                        self.active_alerts[alert_key] = now
                        alerts.append({
                            "type": "oos",
                            "message": f"OUT OF STOCK: '{cell.expected_class}' is missing on {shelf_name} in Bay {slot_id} (Empty for {cell.oos_duration:.1f}s).",
                            "shelf_id": shelf_id,
                            "slot_id": slot_id,
                            "timestamp": now
                        })
                
                # Check for Misplacement warnings
                elif cell.is_occupied and cell.expected_class and cell.current_detection.class_name != cell.expected_class:
                    alert_key = f"misplaced_s{shelf_id}_p{slot_id}"
                    if alert_key not in self.active_alerts or (now - self.active_alerts[alert_key] > self.alert_cooldown):
                        self.active_alerts[alert_key] = now
                        alerts.append({
                            "type": "misplaced",
                            "message": f"PLANOGRAM ERROR: Found '{cell.current_detection.class_name}' on {shelf_name} (Expected '{cell.expected_class}') in Bay {slot_id}.",
                            "shelf_id": shelf_id,
                            "slot_id": slot_id,
                            "timestamp": now
                        })

        # Pack structured shelves list matching active positions
        shelves_list = []
        for r in range(3):
            shelf_id = r + 1
            shelf_name = "Top Shelf - Soda" if r == 0 else ("Middle Shelf - Chips" if r == 1 else "Bottom Shelf - Water")
            y_start = r * 240
            y_end = (r + 1) * 240
            
            slots_list = []
            for c in range(6):
                cell = self.grid_engine.grid[r][c]
                status = "OK" if cell.is_occupied and cell.current_detection.class_name == cell.expected_class else (
                    "MISPLACED" if cell.is_occupied else "OOS"
                )
                slots_list.append({
                    "slot_id": c + 1,
                    "x_range": [c * 213, (c + 1) * 213],
                    "status": status,
                    "current_item": cell.current_detection.class_name if cell.is_occupied else None,
                    "expected": cell.expected_class
                })
                
            shelves_list.append({
                "id": shelf_id,
                "name": shelf_name,
                "y_range": [y_start, y_end],
                "slots": slots_list
            })

        metrics = {
            "occupancy_rate": stats["fill_rate"],
            "total_slots": stats["total_cells"],
            "occupied_slots": stats["occupied"],
            "oos_slots": stats["oos_count"],
            "misplaced_slots": stats["misplacements"],
            "alerts": alerts,
            "shelves": shelves_list
        }
        
        return metrics

    def annotate_frame(self, frame, detections: List[dict]) -> np.ndarray:
        """
        Draws detailed cell compliance grids, time durations, and corner-embedded 
        live OOS thermal heatmaps. Also overlays active YOLO bounding badges.
        """
        # Convert dictionary detections list
        dataclass_dets = []
        for det in detections:
            box = det["box"]
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            area = (box[2] - box[0]) * (box[3] - box[1])
            dataclass_dets.append(Detection(
                bbox=tuple(box),
                confidence=det["confidence"],
                class_id=det["class_id"],
                class_name=det["class"],
                center=(cx, cy),
                area=area
            ))

        # Render PlanogramGrid drawing utilities
        annotated = self.grid_engine.draw_grid(frame, show_labels=True)

        # Superimpose YOLO active labels bounding badges
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_name = det["class"]
            conf = det["confidence"]
            
            # Label tag color mappings
            if cls_name == "Soda":
                badge_color = (0, 0, 220)
            elif cls_name == "Chips":
                badge_color = (0, 180, 220)
            else:
                badge_color = (220, 80, 0)
                
            cv2.rectangle(annotated, (x1, y1), (x2, y2), badge_color, 2)
            label_str = f"{cls_name} {int(conf*100)}%"
            (w, h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 10, y1), badge_color, -1)
            cv2.putText(annotated, label_str, (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated


def generate_demo_planogram(rows: int = 4, cols: int = 6) -> Dict[Tuple[int, int], str]:
    """Generate a demo planogram for testing."""
    products = ['bottle', 'cup', 'bowl', 'apple', 'banana', 'book']
    planogram = {}
    
    for r in range(rows):
        for c in range(cols):
            # Every other cell has expected product
            if (r + c) % 2 == 0:
                planogram[(r, c)] = products[(r * cols + c) % len(products)]
    
    return planogram

