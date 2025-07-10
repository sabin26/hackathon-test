# Sports Analytics Real-Time Dashboard

A beautiful, real-time web dashboard for sports analytics that displays live player tracking, team statistics, ball possession, and event detection as video frames are processed.

## Features

### 🎯 Real-Time Analytics
- **Live Player Tracking**: See players moving on the field in real-time
- **Team Identification**: Automatic team detection with color coding
- **Ball Detection**: Real-time ball position tracking
- **Event Detection**: Live detection of possession, passes, shots, and other events

### 📊 Interactive Visualizations
- **Field Visualization**: Live 2D field view with player and ball positions
- **Possession Chart**: Real-time pie chart showing ball possession statistics
- **Player Heatmap**: Heat map showing player movement patterns
- **Event Timeline**: Chronological display of game events
- **Performance Metrics**: Processing speed and system performance indicators

### 🎨 Beautiful Interface
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-Time Updates**: WebSocket-based live data streaming
- **Professional Styling**: Clean, modern interface with Bootstrap 5
- **Interactive Charts**: Powered by Plotly.js for smooth animations

## Quick Start

### 1. Install Dependencies

```bash
# Install the required packages
pip install -r requirements.txt
```

### 2. Demo Mode (Recommended for First Run)

Start the dashboard with simulated data to see how it works:

```bash
python launch_dashboard.py --mode demo
```

Then open your browser to: **http://localhost:8000**

### 3. Video Processing Mode

To process an actual video file:

```bash
# Make sure your video file is configured in config.yaml
python launch_dashboard.py --mode video --config config.yaml
```

### 4. Advanced Usage

For more control, use the full dashboard runner:

```bash
python run_dashboard.py --host localhost --port 8000 --config config.yaml
```

## Dashboard Components

### 🎮 Header
- **Connection Status**: Shows if the dashboard is connected to the data stream
- **Frame Counter**: Current frame being processed
- **FPS Counter**: Real-time frames per second

### 📈 Left Panel - Live Stats
- **Game Overview**: Number of players detected and ball status
- **Ball Possession**: Pie chart showing possession distribution between teams
- **Recent Events**: Live feed of game events (passes, shots, possession changes)

### 🏟️ Center Panel - Field View
- **Live Field Visualization**: 2D representation of the playing field
- **Player Markers**: Color-coded dots representing players from different teams
- **Ball Marker**: White dot showing ball position
- **Real-Time Movement**: See players and ball move as the video is processed

### 📊 Right Panel - Analytics
- **Player Heatmap**: Shows areas of high player activity
- **Performance Metrics**: Processing speed and system performance
- **Event Timeline**: Chronological chart of events over time

## Configuration

### Video Settings
Edit `config.yaml` to configure your video source:

```yaml
video:
    path: 'your_video.mp4'  # Path to your sports video
    supported_formats: ['.mp4', '.avi', '.mov', '.mkv']
```

### Dashboard Settings
The dashboard server can be configured when launching:

```bash
python run_dashboard.py --host 0.0.0.0 --port 8080  # Make accessible from other devices
```

## Technical Details

### Architecture
- **Backend**: FastAPI with WebSocket support for real-time communication
- **Frontend**: HTML5, CSS3, JavaScript with Bootstrap 5 and Plotly.js
- **Data Processing**: OpenCV, YOLO, and custom analytics pipeline
- **Communication**: WebSocket for low-latency real-time updates

### Data Flow
1. Video frames are processed by the AI analytics pipeline
2. Analytics data is streamed via WebSocket to connected dashboards
3. Frontend JavaScript receives data and updates visualizations in real-time
4. Charts and field view are updated smoothly without page refresh

### Performance
- **Real-Time Updates**: Sub-second latency for live data streaming
- **Efficient Rendering**: Optimized chart updates and DOM manipulation
- **Responsive Design**: Smooth performance on various screen sizes
- **Memory Management**: Automatic cleanup of old data to prevent memory leaks

## Customization

### Adding New Visualizations
1. Add new chart containers to `templates/dashboard.html`
2. Implement chart logic in `static/dashboard.js`
3. Style with custom CSS in `static/dashboard.css`

### Modifying Team Colors
Edit the team color mapping in `static/dashboard.js`:

```javascript
getTeamClass(teamName) {
    if (teamName === 'team_A') return 'team-a';  // Red
    if (teamName === 'team_B') return 'team-b';  // Blue
    return 'unknown';  // Gray
}
```

### Custom Event Types
Add new event types in the CSS for proper styling:

```css
.event-type.your-event {
    background-color: #your-color;
    color: #text-color;
}
```

## Troubleshooting

### Dashboard Not Loading
- Check if the server is running: `curl http://localhost:8000/api/health`
- Verify no other service is using port 8000
- Check the console logs for error messages

### No Data Appearing
- Ensure video processing is running
- Check WebSocket connection status in browser developer tools
- Verify the video file exists and is readable

### Performance Issues
- Reduce video resolution in config.yaml
- Increase frame skip interval for faster processing
- Close other resource-intensive applications

## Browser Compatibility

- **Chrome**: Fully supported (recommended)
- **Firefox**: Fully supported
- **Safari**: Supported with minor limitations
- **Edge**: Fully supported

## Development

### File Structure
```
├── dashboard_server.py      # FastAPI WebSocket server
├── templates/
│   └── dashboard.html       # Main dashboard template
├── static/
│   ├── dashboard.css        # Dashboard styles
│   └── dashboard.js         # Dashboard JavaScript
├── run_dashboard.py         # Full dashboard runner
├── launch_dashboard.py      # Simple launcher
└── ai.py                    # Modified with streaming support
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Sports Analytics system. See the main project license for details.
