/**
 * alerts.js - Alert notifications with sound
 */

class AlertManager {
    constructor() {
        this.container = document.getElementById('toast-container');
        this.audio = document.getElementById('alert-sound');
        this.alertHistory = [];
        this.maxHistory = 20;
        this.soundEnabled = true;
        this.cooldowns = new Map();
        this.cooldownMs = 5000; // 5 seconds between same alert type
        
        this.init();
    }
    
    init() {
        // Create sound toggle if not exists
        this.createSoundToggle();
    }
    
    createSoundToggle() {
        const toggle = document.getElementById('toggle-sound-settings');
        if (!toggle) return;
        
        toggle.addEventListener('click', () => {
            this.soundEnabled = !this.soundEnabled;
            toggle.classList.toggle('active', this.soundEnabled);
            toggle.innerHTML = this.soundEnabled ? '🔊 Sound On' : '🔇 Sound Off';
        });
    }
    
    showAlert(alert) {
        // Check cooldown
        const key = `${alert.type}-${alert.location?.row}-${alert.location?.col}`;
        const lastTime = this.cooldowns.get(key);
        const now = Date.now();
        
        if (lastTime && (now - lastTime) < this.cooldownMs) {
            return; // Skip duplicate
        }
        this.cooldowns.set(key, now);
        
        // Play sound
        if (this.soundEnabled && this.audio) {
            this.audio.currentTime = 0;
            this.audio.play().catch(e => console.log('Audio play failed:', e));
        }
        
        // Only show sliding toast popups for manual user action feedbacks,
        // and show the automatic compliance alerts directly in the feed without the toast popup.
        const isAutoAlert = alert.type === 'oos' || alert.type === 'misplaced';
        
        if (!isAutoAlert) {
            const toast = document.createElement('div');
            toast.className = `toast ${alert.severity || 'info'}`;
            
            const icon = alert.severity === 'critical' ? '🔴' : 
                         alert.severity === 'warning' ? '🟡' : '🔵';
            
            const time = new Date((alert.timestamp || (now / 1000)) * 1000).toLocaleTimeString();
            
            toast.innerHTML = `
                <div class="toast-title">${icon} ${(alert.type || 'info').toUpperCase()}</div>
                <div class="toast-message">${alert.message}</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem;">
                    ${time}
                </div>
            `;
            
            this.container.appendChild(toast);
            
            // Auto remove
            setTimeout(() => {
                toast.style.animation = 'slideInRight 0.3s ease reverse';
                setTimeout(() => toast.remove(), 300);
            }, 5000);
        }
        
        // Add to feed (so the message is always visible in the active alerts list)
        this.addToFeed(alert);
        
        // Store history
        this.alertHistory.push(alert);
        if (this.alertHistory.length > this.maxHistory) {
            this.alertHistory.shift();
        }
    }
    
    addToFeed(alert) {
        const feed = document.getElementById('alert-feed');
        const item = document.createElement('div');
        item.className = `alert-item ${alert.severity}`;
        
        const time = new Date(alert.timestamp * 1000).toLocaleTimeString();
        
        item.innerHTML = `
            <div>${alert.message}</div>
            <div class="time">${time}</div>
        `;
        
        feed.insertBefore(item, feed.firstChild);
        
        // Keep only last 10
        while (feed.children.length > 10) {
            feed.removeChild(feed.lastChild);
        }
    }
    
    getHistory() {
        return this.alertHistory;
    }
}

// Initialize
const alertManager = new AlertManager();
window.alertManager = alertManager;
