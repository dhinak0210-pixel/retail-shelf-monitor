/**
 * heatmap.js - OOS frequency heatmap visualization
 */

class HeatmapManager {
    constructor() {
        this.canvas = document.getElementById('heatmap-overlay');
        this.ctx = this.canvas.getContext('2d');
        this.isActive = false;
        this.gridRows = 4;
        this.gridCols = 6;
        this.oosData = new Map(); // cell key -> accumulated OOS time
        
        this.init();
    }
    
    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }
    
    resize() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
    }
    
    update(gridCells) {
        if (!this.isActive) return;
        
        const w = this.canvas.width;
        const h = this.canvas.height;
        const cellW = w / this.gridCols;
        const cellH = h / this.gridRows;
        
        // Clear
        this.ctx.clearRect(0, 0, w, h);
        
        // Accumulate OOS data
        gridCells.forEach(cell => {
            const key = `${cell.row}-${cell.col}`;
            if (!cell.occupied && cell.expected) {
                const current = this.oosData.get(key) || 0;
                this.oosData.set(key, current + 1);
            }
        });
        
        // Find max for normalization
        const maxVal = Math.max(...this.oosData.values(), 1);
        
        // Draw heatmap cells
        this.oosData.forEach((value, key) => {
            const [row, col] = key.split('-').map(Number);
            const intensity = Math.min(value / maxVal, 1);
            
            const x = col * cellW;
            const y = row * cellH;
            
            // Color from blue (low) to red (high)
            const r = Math.floor(255 * intensity);
            const g = Math.floor(255 * (1 - intensity) * 0.3);
            const b = Math.floor(255 * (1 - intensity));
            
            this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.5)`;
            this.ctx.fillRect(x, y, cellW, cellH);
            
            // Border
            this.ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(x, y, cellW, cellH);
        });
        
        // Draw legend
        this.drawLegend(w, h);
    }
    
    drawLegend(w, h) {
        const legendW = 150;
        const legendH = 15;
        const x = w - legendW - 20;
        const y = h - 40;
        
        // Gradient bar
        const gradient = this.ctx.createLinearGradient(x, 0, x + legendW, 0);
        gradient.addColorStop(0, 'rgba(0, 0, 255, 0.5)');
        gradient.addColorStop(0.5, 'rgba(255, 255, 0, 0.5)');
        gradient.addColorStop(1, 'rgba(255, 0, 0, 0.5)');
        
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(x, y, legendW, legendH);
        
        // Labels
        this.ctx.fillStyle = 'white';
        this.ctx.font = '10px sans-serif';
        this.ctx.fillText('Low OOS', x, y - 5);
        this.ctx.fillText('High OOS', x + legendW - 50, y - 5);
    }
    
    reset() {
        this.oosData.clear();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

// Initialize
const heatmapManager = new HeatmapManager();
window.heatmapManager = heatmapManager;
