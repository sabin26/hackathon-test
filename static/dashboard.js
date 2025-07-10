/**
 * Sports Analytics Dashboard JavaScript
 * Handles real-time data visualization and WebSocket communication
 */

class SportsAnalyticsDashboard {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.frameCount = 0;
        this.lastUpdateTime = Date.now();
        this.fpsCounter = 0;
        
        // Chart instances
        this.possessionChart = null;
        this.fieldChart = null;
        this.heatmapChart = null;
        this.timelineChart = null;
        
        // Data storage
        this.gameStats = {};
        this.playerPositions = {};
        this.events = [];
        
        this.init();
    }
    
    init() {
        this.connectWebSocket();
        this.initializeCharts();
        this.setupEventListeners();
        
        // Start FPS counter
        setInterval(() => this.updateFPS(), 1000);
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.updateConnectionStatus(true);
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus(false);
                
                // Attempt to reconnect after 3 seconds
                setTimeout(() => this.connectWebSocket(), 3000);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    handleMessage(message) {
        switch (message.type) {
            case 'frame_update':
                this.updateDashboard(message.data, message.stats);
                break;
            case 'initial_data':
                this.updateDashboard(message.data, message.stats);
                break;
            case 'stats_update':
                this.updateStats(message.stats);
                break;
            case 'pong':
                // Handle ping/pong for connection health
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    }
    
    updateDashboard(frameData, stats) {
        this.frameCount++;
        this.fpsCounter++;
        
        // Update frame counter
        document.getElementById('frame-counter').textContent = `Frame: ${this.frameCount}`;
        
        // Update game stats
        this.gameStats = stats;
        this.updateGameOverview(stats);
        this.updatePossessionChart(stats);
        this.updateFieldVisualization(frameData);
        this.updateHeatmap(stats);
        this.updateEvents(stats);
        this.updatePerformanceMetrics(stats);
        this.updateTimeline(stats);
    }
    
    updateGameOverview(stats) {
        // Update player count
        document.getElementById('players-count').textContent = stats.players_detected || 0;
        
        // Update ball status
        const ballStatus = document.getElementById('ball-status');
        if (stats.ball_detected) {
            ballStatus.innerHTML = '<i class="fas fa-check text-success"></i>';
        } else {
            ballStatus.innerHTML = '<i class="fas fa-times text-danger"></i>';
        }
    }
    
    updatePossessionChart(stats) {
        if (!this.possessionChart) return;
        
        const possessionData = stats.possession_stats || {};
        const total = Object.values(possessionData).reduce((sum, val) => sum + val, 0);
        
        if (total === 0) return;
        
        const data = [{
            values: Object.values(possessionData),
            labels: Object.keys(possessionData).map(key => 
                key === 'none' ? 'No Possession' : `Team ${key.toUpperCase()}`
            ),
            type: 'pie',
            hole: 0.4,
            marker: {
                colors: ['#ff4444', '#4444ff', '#888888']
            }
        }];
        
        const layout = {
            margin: { t: 0, b: 0, l: 0, r: 0 },
            showlegend: true,
            legend: { orientation: 'h', y: -0.1 },
            font: { size: 10 }
        };
        
        Plotly.react('possession-chart', data, layout, { displayModeBar: false });
    }
    
    updateFieldVisualization(frameData) {
        if (!frameData || !frameData.objects) return;
        
        const fieldContainer = document.getElementById('field-visualization');
        
        // Clear existing markers
        fieldContainer.querySelectorAll('.player-marker, .ball-marker').forEach(el => el.remove());
        
        // Add player markers
        frameData.objects.forEach(obj => {
            if (obj.type === 'person' && obj.pos_pitch) {
                this.addPlayerMarker(fieldContainer, obj);
            } else if (obj.type === 'sports ball' && obj.pos_pitch) {
                this.addBallMarker(fieldContainer, obj);
            }
        });
    }
    
    addPlayerMarker(container, player) {
        const marker = document.createElement('div');
        marker.className = `player-marker ${this.getTeamClass(player.team_name)}`;
        
        // Position on field (assuming field dimensions)
        const fieldWidth = container.offsetWidth;
        const fieldHeight = container.offsetHeight;
        const x = (player.pos_pitch[0] / 100) * fieldWidth; // Assuming 100m field width
        const y = (player.pos_pitch[1] / 60) * fieldHeight;  // Assuming 60m field height
        
        marker.style.left = `${Math.max(0, Math.min(x, fieldWidth - 12))}px`;
        marker.style.top = `${Math.max(0, Math.min(y, fieldHeight - 12))}px`;
        
        // Add tooltip
        marker.title = `Player ${player.id} - ${player.team_name || 'Unknown'} - Jersey: ${player.jersey_number || 'N/A'}`;
        
        container.appendChild(marker);
    }
    
    addBallMarker(container, ball) {
        const marker = document.createElement('div');
        marker.className = 'ball-marker';
        
        const fieldWidth = container.offsetWidth;
        const fieldHeight = container.offsetHeight;
        const x = (ball.pos_pitch[0] / 100) * fieldWidth;
        const y = (ball.pos_pitch[1] / 60) * fieldHeight;
        
        marker.style.left = `${Math.max(0, Math.min(x, fieldWidth - 8))}px`;
        marker.style.top = `${Math.max(0, Math.min(y, fieldHeight - 8))}px`;
        
        marker.title = 'Ball';
        
        container.appendChild(marker);
    }
    
    getTeamClass(teamName) {
        if (!teamName || teamName === 'unknown' || teamName === 'none') {
            return 'unknown';
        }
        return teamName.toLowerCase().includes('a') ? 'team-a' : 'team-b';
    }
    
    updateHeatmap(stats) {
        // Simple heatmap visualization
        // In a real implementation, you'd create a proper heatmap
        if (!this.heatmapChart) {
            this.initializeHeatmap();
        }
        
        // Update with player position data
        const positions = stats.player_positions || {};
        const heatmapData = this.generateHeatmapData(positions);
        
        if (heatmapData.length > 0) {
            Plotly.react('heatmap-chart', heatmapData, {
                margin: { t: 0, b: 0, l: 0, r: 0 },
                xaxis: { visible: false },
                yaxis: { visible: false }
            }, { displayModeBar: false });
        }
    }
    
    generateHeatmapData(positions) {
        // Generate heatmap data from player positions
        const data = [];
        
        Object.values(positions).forEach(playerPositions => {
            if (playerPositions.length > 0) {
                const x = playerPositions.map(pos => pos.x);
                const y = playerPositions.map(pos => pos.y);
                
                data.push({
                    x: x,
                    y: y,
                    type: 'scatter',
                    mode: 'markers',
                    marker: {
                        size: 4,
                        opacity: 0.6,
                        color: 'red'
                    }
                });
            }
        });
        
        return data;
    }
    
    updateEvents(stats) {
        const eventsContainer = document.getElementById('events-list');
        const events = stats.events || [];
        
        if (events.length === 0) {
            eventsContainer.innerHTML = '<div class="text-muted text-center p-3">No events yet...</div>';
            return;
        }
        
        // Show only the latest 10 events
        const recentEvents = events.slice(-10).reverse();
        
        eventsContainer.innerHTML = recentEvents.map(event => `
            <div class="event-item">
                <div class="event-time">${this.formatTimestamp(event.timestamp)}</div>
                <div class="event-description">
                    Player ${event.player_id} - ${event.team_name}
                </div>
                <span class="event-type ${event.event_type.toLowerCase()}">${event.event_type}</span>
            </div>
        `).join('');
    }
    
    updatePerformanceMetrics(stats) {
        const metrics = stats.performance_metrics || {};
        
        document.getElementById('processing-fps').textContent = 
            Math.round(metrics.fps || 0);
        document.getElementById('processing-time').textContent = 
            `${Math.round(metrics.processing_time || 0)}ms`;
    }
    
    updateTimeline(stats) {
        // Simple timeline chart
        const events = stats.events || [];
        
        if (events.length === 0) return;
        
        const timelineData = [{
            x: events.map(e => e.timestamp),
            y: events.map((e, i) => i),
            text: events.map(e => `${e.event_type} - Player ${e.player_id}`),
            mode: 'markers+lines',
            type: 'scatter',
            marker: { size: 8 }
        }];
        
        const layout = {
            margin: { t: 10, b: 30, l: 30, r: 10 },
            xaxis: { title: 'Time (s)' },
            yaxis: { title: 'Events' },
            showlegend: false
        };
        
        Plotly.react('timeline-chart', timelineData, layout, { displayModeBar: false });
    }
    
    initializeCharts() {
        // Initialize empty charts
        this.initializePossessionChart();
        this.initializeHeatmap();
    }
    
    initializePossessionChart() {
        const data = [{
            values: [1],
            labels: ['No Data'],
            type: 'pie',
            hole: 0.4,
            marker: { colors: ['#e9ecef'] }
        }];
        
        const layout = {
            margin: { t: 0, b: 0, l: 0, r: 0 },
            showlegend: false,
            font: { size: 10 }
        };
        
        Plotly.newPlot('possession-chart', data, layout, { displayModeBar: false });
        this.possessionChart = true;
    }
    
    initializeHeatmap() {
        const data = [{
            x: [0],
            y: [0],
            type: 'scatter',
            mode: 'markers',
            marker: { size: 1, opacity: 0 }
        }];
        
        Plotly.newPlot('heatmap-chart', data, {
            margin: { t: 0, b: 0, l: 0, r: 0 },
            xaxis: { visible: false },
            yaxis: { visible: false }
        }, { displayModeBar: false });
        this.heatmapChart = true;
    }
    
    setupEventListeners() {
        // Handle window resize
        window.addEventListener('resize', () => {
            this.resizeCharts();
        });
    }
    
    resizeCharts() {
        // Resize all Plotly charts
        Plotly.Plots.resize('possession-chart');
        Plotly.Plots.resize('heatmap-chart');
        Plotly.Plots.resize('timeline-chart');
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (connected) {
            statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>Connected';
            statusElement.className = 'badge bg-success me-3';
        } else {
            statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>Disconnected';
            statusElement.className = 'badge bg-danger me-3';
        }
    }
    
    updateFPS() {
        document.getElementById('fps-counter').textContent = `FPS: ${this.fpsCounter}`;
        this.fpsCounter = 0;
    }
    
    formatTimestamp(timestamp) {
        const minutes = Math.floor(timestamp / 60);
        const seconds = Math.floor(timestamp % 60);
        return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new SportsAnalyticsDashboard();
});
