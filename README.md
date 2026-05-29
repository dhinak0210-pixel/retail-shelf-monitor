---
title: Retail Shelf Monitor
emoji: 🛒
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 🛒 Aether-Shelf AI — Real-Time Retail Shelf Monitor

A high-fidelity, real-time AI retail shelf compliance and telemetry monitor utilizing a custom YOLOv8 model, designed with a luxury skeuomorphic wood-and-brass manager's dashboard.

## 🌟 Core Features
- **📦 Planogram Compliance Matrix**: Real-time object recognition grid tracking Soda, Chips, and Water Bottles.
- **🚶 Shopper Traffic overlay**: Interactive Dwell Pathways and column heatmap analytics.
- **🛡️ Physical Security**: Laplacian variance blur alarm and illuminance occlusion detectors.
- **🎛️ Campaign Manager**: Dynamic campaign templates to change shelf targets instantly.
- **📊 Shift SLA Compliance Auditor**: Multi-page printable shift restock reports (MTTR KPIs).

## 🚀 Running Locally

Build and run the container on your local machine:
```bash
docker build -t shelf-monitor .
docker run -p 7860:7860 --device=/dev/video0:/dev/video0 shelf-monitor
```
