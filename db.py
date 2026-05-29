"""
db.py - SQLite database manager for persisting shelf compliance logs and historical stats
"""

import sqlite3
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple

DB_PATH = "data/compliance.db"

def init_db():
    """Initializes SQLite database schemas if not present."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table 1: Compliance Alerts History (OOS, Misplacements, OK states)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS compliance_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            datetime_str TEXT,
            event_type TEXT, -- 'oos', 'misplaced', 'ok'
            shelf_id INTEGER,
            slot_id INTEGER,
            expected_class TEXT,
            current_class TEXT,
            duration REAL
        )
    """)
    
    # Table 2: Hourly Performance Metric Records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hourly_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            datetime_str TEXT,
            occupancy_rate REAL,
            oos_count INTEGER,
            misplaced_count INTEGER,
            compliance_rate REAL
        )
    """)
    
    conn.commit()
    conn.close()

def log_compliance_event(event_type: str, shelf_id: int, slot_id: int, expected: str, current: str, duration: float = 0.0):
    """Inserts a single planogram discrepancy alert event."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = time.time()
    dt_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO compliance_events (timestamp, datetime_str, event_type, shelf_id, slot_id, expected_class, current_class, duration)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (now, dt_str, event_type, shelf_id, slot_id, expected, current, duration))
    
    conn.commit()
    conn.close()

def log_hourly_stats(occupancy_rate: float, oos_count: int, misplaced_count: int, compliance_rate: float):
    """Persists aggregated telemetry analytics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = time.time()
    dt_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO hourly_stats (timestamp, datetime_str, occupancy_rate, oos_count, misplaced_count, compliance_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (now, dt_str, occupancy_rate, oos_count, misplaced_count, compliance_rate))
    
    conn.commit()
    conn.close()

def get_recent_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves list of latest compliance alert logs."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM compliance_events ORDER BY id DESC LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def export_compliance_csv_data() -> Tuple[str, List[List[str]]]:
    """Generates standard CSV headers and rows representing the complete history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT datetime_str, event_type, shelf_id, slot_id, expected_class, current_class, duration 
        FROM compliance_events 
        ORDER BY id DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    headers = ["Timestamp", "Event Type", "Shelf ID", "Slot ID", "Expected SKU", "Current Detection", "OOS Duration (s)"]
    csv_rows = [headers]
    for row in rows:
        csv_rows.append([str(val) if val is not None else "" for val in row])
        
    return "compliance_report.csv", csv_rows
