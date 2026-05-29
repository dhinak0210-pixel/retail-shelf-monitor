/**
 * stats.js - WebSocket connection and stats handling
 */

class StatsManager {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 3000;
        this.isConnected = false;
        this.statsHistory = [];
        this.maxHistory = 100;
        
        this.init();
    }
    
    init() {
        this.connect();
        this.setupControls();
    }
    
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/stats`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.updateConnectionStatus(true);
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'ping') return;
            this.handleStats(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus(false);
            setTimeout(() => this.connect(), this.reconnectInterval);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    handleStats(data) {
        // Update FPS and inference time
        document.getElementById('fps-display').textContent = `FPS: ${data.fps || '--'}`;
        document.getElementById('inference-time').textContent = `Inference: ${data.inference_ms || '--'} ms`;
        
        // Update shelf stats
        const stats = data.shelf_stats || {};
        document.getElementById('fill-rate').textContent = `${stats.fill_rate || 0}%`;
        document.getElementById('oos-count').textContent = stats.oos_count || 0;
        document.getElementById('compliance').textContent = `${stats.planogram_compliance || 0}%`;
        document.getElementById('misplacements').textContent = stats.misplacements || 0;
        
        // Update fill bar
        const fillBar = document.getElementById('fill-bar');
        fillBar.style.width = `${stats.fill_rate || 0}%`;
        fillBar.style.background = stats.fill_rate > 80 ? 'var(--success)' : 
                                   stats.fill_rate > 50 ? 'var(--warning)' : 'var(--danger)';
        
        // Update detections list
        this.updateDetections(data.detections || []);
        
        // Update stockout forecasting predictions
        this.updateForecastProjections(data.forecasts || []);
        
        // Update dwell pathway visual if enabled
        if (this.dwellActive) {
            this.updateDwellVisualization(data.shopper_tracks || [], data.dwell_times || {});
        }
        
        // Update grid visual
        this.updateGridVisual(data.grid_cells || []);
        
        // Handle alerts
        if (data.alerts && data.alerts.length > 0) {
            data.alerts.forEach(alert => {
                window.alertManager?.showAlert(alert);
            });
        }
        
        // Update heatmap if active
        if (window.heatmapManager?.isActive) {
            window.heatmapManager.update(data.grid_cells);
        }
        
        // Store history
        this.statsHistory.push(data);
        if (this.statsHistory.length > this.maxHistory) {
            this.statsHistory.shift();
        }

        // Update camera health alert banner
        const healthBanner = document.getElementById('camera-health-warning');
        if (healthBanner && data.camera_health) {
            if (data.camera_health.is_tampered) {
                healthBanner.style.display = 'block';
                healthBanner.innerText = `⚠️ CAMERA HEALTH ALERT: ${data.camera_health.reason}`;
            } else {
                healthBanner.style.display = 'none';
            }
        }
    }
    
    updateDetections(detections) {
        const list = document.getElementById('detections-list');
        list.innerHTML = '';
        
        detections.slice(0, 5).forEach(det => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span>${det.class_name}</span>
                <span class="conf">${(det.confidence * 100).toFixed(0)}%</span>
            `;
            list.appendChild(li);
        });
    }
    
    updateGridVisual(cells) {
        const container = document.getElementById('grid-visual');
        
        // Only rebuild if first time
        if (container.children.length === 0) {
            container.innerHTML = '';
            cells.forEach(cell => {
                const div = document.createElement('div');
                div.className = 'grid-cell';
                div.dataset.row = cell.row;
                div.dataset.col = cell.col;
                div.title = `Row ${cell.row}, Col ${cell.col}`;
                
                // Add click listener to trigger the context target selector menu
                div.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const menu = document.getElementById('grid-context-menu');
                    this.activeRow = parseInt(cell.row);
                    this.activeCol = parseInt(cell.col);
                    
                    // Place context selector menu absolute coordinates
                    const rect = div.getBoundingClientRect();
                    menu.style.left = `${rect.left + window.scrollX}px`;
                    menu.style.top = `${rect.bottom + window.scrollY + 5}px`;
                    menu.style.display = 'flex';
                });
                
                container.appendChild(div);
            });
        }
        
        // Update classes and target label overlays
        cells.forEach(cell => {
            const el = container.querySelector(`[data-row="${cell.row}"][data-col="${cell.col}"]`);
            if (!el) return;
            
            el.className = 'grid-cell';
            if (cell.occupied) {
                el.classList.add('occupied');
            } else if (cell.expected) {
                el.classList.add('empty');
            }
            if (cell.expected) {
                el.classList.add('expected');
            }
            
            // Render text targets inside HUD visualizer
            const expectedStr = cell.expected ? cell.expected.substring(0, 4) : 'Empty';
            el.innerHTML = `<span style="font-size: 0.65rem; font-weight: bold; pointer-events: none;">${expectedStr}</span>`;
        });
    }
    
    updateConnectionStatus(connected) {
        const status = document.getElementById('connection-status');
        status.className = `status ${connected ? 'connected' : 'disconnected'}`;
        status.textContent = connected ? '● Connected' : '● Disconnected';
    }
    
    setupControls() {
        // Toggle grid overlay
        document.getElementById('toggle-grid').addEventListener('click', (e) => {
            e.target.classList.toggle('active');
            // Grid is baked into video feed, this controls UI grid visual
            document.getElementById('grid-visual').style.opacity = 
                e.target.classList.contains('active') ? '1' : '0.3';
        });
        
        // Toggle heatmap
        document.getElementById('toggle-heatmap').addEventListener('click', (e) => {
            e.target.classList.toggle('active');
            const overlay = document.getElementById('heatmap-overlay');
            overlay.classList.toggle('active');
            window.heatmapManager.isActive = e.target.classList.contains('active');
        });
        // Toggle detections (reloads video to apply server-side)
        document.getElementById('toggle-detections').addEventListener('click', (e) => {
            e.target.classList.toggle('active');
        });

        // Toggle dwell foot-traffic mapping HUD
        this.dwellActive = false;
        document.getElementById('toggle-dwell').addEventListener('click', (e) => {
            e.target.classList.toggle('active');
            this.dwellActive = e.target.classList.contains('active');
            if (!this.dwellActive) {
                const canvas = document.getElementById('dwell-canvas');
                if (canvas) {
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
            }
        });

        // ROI Drag Bounding Box Drawer Controls
        const toggleRoiBtn = document.getElementById('toggle-roi');
        const roiCanvas = document.getElementById('roi-canvas');
        const roiInstruction = document.getElementById('roi-instruction');
        const videoFeed = document.getElementById('video-feed');

        toggleRoiBtn.addEventListener('click', () => {
            const active = toggleRoiBtn.classList.toggle('active');
            if (active) {
                // Sync dimensions of the draw layer with active visual boundaries
                roiCanvas.width = videoFeed.clientWidth;
                roiCanvas.height = videoFeed.clientHeight;
                roiCanvas.style.pointerEvents = 'auto';
                roiCanvas.style.cursor = 'crosshair';
                roiInstruction.style.display = 'block';
            } else {
                roiCanvas.style.pointerEvents = 'none';
                roiCanvas.style.cursor = 'default';
                roiInstruction.style.display = 'none';
                const ctx = roiCanvas.getContext('2d');
                ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
            }
        });

        let drawing = false;
        let startX = 0;
        let startY = 0;

        roiCanvas.addEventListener('mousedown', (e) => {
            drawing = true;
            startX = e.offsetX;
            startY = e.offsetY;
        });

        roiCanvas.addEventListener('mousemove', (e) => {
            if (!drawing) return;
            const ctx = roiCanvas.getContext('2d');
            ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
            ctx.strokeStyle = '#00ffcc';
            ctx.lineWidth = 3;
            ctx.setLineDash([6, 6]);
            ctx.strokeRect(startX, startY, e.offsetX - startX, e.offsetY - startY);
        });

        const finishDrawing = (e) => {
            if (!drawing) return;
            drawing = false;
            
            const ctx = roiCanvas.getContext('2d');
            ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
            
            toggleRoiBtn.classList.remove('active');
            roiCanvas.style.pointerEvents = 'none';
            roiInstruction.style.display = 'none';

            // Scale to native video feed coordinates
            const scaleX = 1280 / roiCanvas.width;
            const scaleY = 720 / roiCanvas.height;

            const x_min = Math.round(Math.min(startX, e.offsetX) * scaleX);
            const y_min = Math.round(Math.min(startY, e.offsetY) * scaleY);
            const x_max = Math.round(Math.max(startX, e.offsetX) * scaleX);
            const y_max = Math.round(Math.max(startY, e.offsetY) * scaleY);

            if (x_max - x_min < 25 || y_max - y_min < 25) return;

            fetch('/api/grid/roi', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x_min, y_min, x_max, y_max })
            }).then(res => res.json()).then(data => {
                window.alertManager?.showAlert({
                    type: 'success',
                    message: 'Aligned Region of Interest (ROI) custom geometry rules!'
                });
            });
        };

        roiCanvas.addEventListener('mouseup', finishDrawing);
        roiCanvas.addEventListener('mouseleave', finishDrawing);

        // Hide context menu on body clicks
        document.addEventListener('click', () => {
            document.getElementById('grid-context-menu').style.display = 'none';
        });

        // Grid selection target context options click
        document.querySelectorAll('.context-option').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const selectedSku = btn.dataset.sku;
                const menu = document.getElementById('grid-context-menu');
                menu.style.display = 'none';

                fetch('/api/planogram', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        layout: [{
                            row: this.activeRow,
                            col: this.activeCol,
                            expected_class: selectedSku === "Empty" ? null : selectedSku
                        }]
                    })
                }).then(res => res.json()).then(data => {
                    window.alertManager?.showAlert({
                        type: 'info',
                        message: `Updated Slot [${this.activeRow + 1}, ${this.activeCol + 1}] Target SKU to: ${selectedSku}`
                    });
                });
            });
        });

        // Model Swapper Option Controls
        document.getElementById('model-selector').addEventListener('change', (e) => {
            const size = e.target.value;
            fetch('/api/model/swap', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_size: size })
            }).then(res => res.json()).then(data => {
                window.alertManager?.showAlert({
                    type: 'info',
                    message: `Successfully swapped detector to: YOLOv8 ${size.toUpperCase()}`
                });
            });
        });

        // Adaptive Settings Slider Controllers
        const fpsSlider = document.getElementById('fps-slider');
        const resSlider = document.getElementById('res-slider');

        const updateAdaptiveSettings = () => {
            const fps = parseFloat(fpsSlider.value);
            const scale = parseFloat(resSlider.value);

            document.getElementById('fps-val').textContent = `${fps} FPS`;
            document.getElementById('res-val').textContent = `${(scale * 100).toFixed(0)}% (${scale >= 1.0 ? '720p' : scale >= 0.7 ? '480p' : '360p'})`;

            fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    fps_limit: fps,
                    quality: 85,
                    resolution_scale: scale
                })
            });
        };

        fpsSlider.addEventListener('input', updateAdaptiveSettings);
        resSlider.addEventListener('input', updateAdaptiveSettings);

        // Planogram Campaign Templates UI Management
        const templateSelector = document.getElementById('template-selector');
        const loadTemplateBtn = document.getElementById('btn-load-template');
        const saveTemplateBtn = document.getElementById('btn-save-template');
        const newTemplateInput = document.getElementById('new-template-name');

        const refreshTemplatesList = () => {
            fetch('/api/planogram/templates')
                .then(res => res.json())
                .then(data => {
                    if (data.templates) {
                        templateSelector.innerHTML = '';
                        data.templates.forEach(t => {
                            const opt = document.createElement('option');
                            opt.value = t;
                            opt.textContent = t;
                            templateSelector.appendChild(opt);
                        });
                    }
                });
        };

        loadTemplateBtn.addEventListener('click', () => {
            const name = templateSelector.value;
            if (!name) return;

            fetch('/api/planogram/templates/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name })
            })
            .then(res => res.json())
            .then(data => {
                window.alertManager?.showAlert({
                    type: 'success',
                    message: data.message || `Loaded planogram campaign: ${name}`
                });
                const visual = document.getElementById('grid-visual');
                if (visual) visual.innerHTML = '';
            });
        });

        saveTemplateBtn.addEventListener('click', () => {
            const name = newTemplateInput.value.trim();
            if (!name) {
                window.alertManager?.showAlert({
                    type: 'error',
                    message: 'Please enter a valid template name'
                });
                return;
            }

            fetch('/api/planogram/templates/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name })
            })
            .then(res => res.json())
            .then(data => {
                window.alertManager?.showAlert({
                    type: 'success',
                    message: data.message || `Saved planogram template: ${name}`
                });
                newTemplateInput.value = '';
                refreshTemplatesList();
            });
        });

        // Initialize templates dropdown list
        refreshTemplatesList();
    }

    updateForecastProjections(forecasts) {
        const feed = document.getElementById('forecast-feed');
        if (!feed) return;
        feed.innerHTML = '';

        if (!forecasts || forecasts.length === 0) {
            feed.innerHTML = `<div style="font-size: 0.8rem; color: var(--text-secondary); text-align: center; padding: 0.5rem 0;">Analyzing consumption rate...</div>`;
            return;
        }

        forecasts.forEach(f => {
            const div = document.createElement('div');
            div.style.display = 'flex';
            div.style.flexDirection = 'column';
            div.style.padding = '0.5rem';
            div.style.borderRadius = '4px';
            div.style.fontSize = '0.8rem';
            div.style.background = 'var(--bg-tertiary)';
            div.style.border = '1px solid var(--border)';
            div.style.marginBottom = '0.25rem';
            
            let badgeBg = 'rgba(76, 175, 80, 0.2)';
            let badgeColor = '#4CAF50';
            let borderStyle = '1px solid rgba(76, 175, 80, 0.4)';
            
            if (f.risk === 'critical' || f.risk === 'empty') {
                badgeBg = 'rgba(244, 67, 54, 0.2)';
                badgeColor = '#F44336';
                borderStyle = '1px solid rgba(244, 67, 54, 0.4)';
            } else if (f.risk === 'warning') {
                badgeBg = 'rgba(255, 152, 0, 0.2)';
                badgeColor = '#FF9800';
                borderStyle = '1px solid rgba(255, 152, 0, 0.4)';
            }

            div.style.borderLeft = `4px solid ${badgeColor}`;

            div.innerHTML = `
                <div style="display: flex; justify-content: space-between; font-weight: bold; margin-bottom: 0.25rem;">
                    <span>📦 ${f.sku}</span>
                    <span style="background: ${badgeBg}; color: ${badgeColor}; padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.7rem; border: ${borderStyle}; text-transform: uppercase;">
                        ${f.risk}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-secondary);">
                    <span>Stock: ${f.stock} items</span>
                    <span>${f.message}</span>
                </div>
            `;
            feed.appendChild(div);
        });
    }

    updateDwellVisualization(tracks, dwellTimes) {
        const canvas = document.getElementById('dwell-canvas');
        if (!canvas) return;
        
        const videoFeed = document.getElementById('video-feed');
        canvas.width = videoFeed.clientWidth;
        canvas.height = videoFeed.clientHeight;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const scaleX = canvas.width / 1280;
        const scaleY = canvas.height / 720;

        // 1. Draw Dwell Heatmap columns overlay
        const colsCount = 6;
        const colWidth = canvas.width / colsCount;
        
        ctx.save();
        for (let c = 0; c < colsCount; c++) {
            const dwellSecs = parseFloat(dwellTimes[c.toString()] || 0.0);
            if (dwellSecs > 0) {
                const alpha = Math.min(dwellSecs / 15.0, 0.45);
                ctx.fillStyle = `rgba(168, 85, 247, ${alpha})`; // Purple neon fill
                ctx.fillRect(c * colWidth, 0, colWidth, canvas.height);
                
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 10px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(`👤 ${dwellSecs.toFixed(1)}s`, c * colWidth + colWidth/2, canvas.height - 25);
            }
        }
        ctx.restore();

        // 2. Draw shopper trajectory pathways
        if (tracks && tracks.length >= 2) {
            ctx.beginPath();
            ctx.strokeStyle = '#a855f7';
            ctx.lineWidth = 4;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
            ctx.setLineDash([4, 6]);

            ctx.moveTo(tracks[0].x * scaleX, tracks[0].y * scaleY);
            for (let i = 1; i < tracks.length; i++) {
                ctx.lineTo(tracks[i].x * scaleX, tracks[i].y * scaleY);
            }
            ctx.stroke();

            // Glowing shopper indicator
            const latest = tracks[tracks.length - 1];
            ctx.beginPath();
            ctx.arc(latest.x * scaleX, latest.y * scaleY, 8, 0, 2 * Math.PI);
            ctx.fillStyle = '#c084fc';
            ctx.shadowBlur = 15;
            ctx.shadowColor = '#a855f7';
            ctx.fill();
        }
    }
    
    getHistory() {
        return this.statsHistory;
    }
}

// Initialize
const statsManager = new StatsManager();
