# 🏈 Sports Analytics System

A comprehensive real-time sports analytics platform that uses computer vision and machine learning to analyze sports videos, track players, identify teams, detect jersey numbers, and provide detailed performance insights.

## 🎯 What This Project Does

This system transforms raw sports video footage into actionable analytics by:

-   **🎥 Video Analysis**: Processes sports videos frame-by-frame using advanced computer vision
-   **👥 Player Tracking**: Uses YOLO v13 object detection to track players throughout the game
-   **🏃‍♂️ Team Identification**: Automatically classifies players into teams using machine learning clustering
-   **🔢 Jersey Number Detection**: Recognizes jersey numbers using OCR technology
-   **📍 Spatial Mapping**: Maps player positions from video coordinates to real pitch coordinates using homography
-   **📊 Real-time Dashboard**: Provides live analytics through an interactive web interface
-   **⚡ Performance Metrics**: Tracks player movements, team formations, and game statistics

## 🏗️ Architecture

The system is built with a clean, modular architecture:

```
├── src/
│   ├── core/           # Core business logic (Config, Homography)
│   ├── models/         # AI/ML models (YOLO, Segmentation)
│   ├── analytics/      # Analytics processing (Team ID, Jersey Detection)
│   ├── api/           # Web server & API (Dashboard, WebSocket)
│   └── utils/         # Utility functions (Logging, Video, Performance)
├── config/            # Configuration files
├── static/            # Web assets (CSS, JavaScript)
├── templates/         # HTML templates
├── tests/             # Test suite
└── uploads/           # Video upload directory
```

## 🚀 Quick Start

### Prerequisites

-   Python 3.8+
-   pip or uv package manager
-   At least 4GB RAM (8GB+ recommended)
-   GPU support optional but recommended for better performance

### Installation

1. **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd hackathon-test
    ```

2. **Install dependencies**:

    ```bash
    # Using pip
    pip install -r requirements.txt

    # Or using uv (faster)
    uv add -r requirements.txt
    ```

### Running the Application

#### Option 1: Web Dashboard (Recommended)

Start the interactive web dashboard:

```bash
python main.py
```

Then open your browser to: **http://localhost:8000**

**Features:**

-   Upload video files for analysis
-   Real-time analytics streaming
-   Interactive data visualization
-   Team and player statistics
-   Performance metrics dashboard

#### Option 2: Command Line Analysis

Analyze a video directly from command line:

```bash
python analyze_video.py path/to/your/video.mp4 --output results.csv
```

#### Option 3: Legacy Dashboard

Use the original dashboard interface:

```bash
python run_dashboard.py
```

### Configuration

Edit `config/config.yaml` to customize:

```yaml
video:
    supported_formats: ['.mp4', '.avi', '.mov', '.mkv']
    min_resolution: [320, 240]
    max_resolution: [4096, 2160]

models:
    yolo_path: 'yolov8n.pt'
    jersey_yolo_path: null # Path to custom jersey detection model

processing:
    team_n_clusters: 3
    frame_skip_interval: 2
    enable_jersey_detection: true
    enable_ocr: true
    enable_homography: true
```

## 📋 Usage Examples

### Web Dashboard Workflow

1. **Start the server**: `python main.py`
2. **Open browser**: Navigate to `http://localhost:8000`
3. **Upload video**: Click "Upload Video" and select your sports video
4. **View analytics**: Watch real-time analysis results as they stream in
5. **Export data**: Download results as CSV for further analysis

### Command Line Workflow

```bash
# Basic analysis
python analyze_video.py game_footage.mp4

# Custom output location
python analyze_video.py game_footage.mp4 --output detailed_analysis.csv

# Custom configuration
python analyze_video.py game_footage.mp4 --config custom_config.yaml
```

## 🔧 Advanced Features

### Real-time Streaming

The dashboard supports real-time analytics streaming via WebSocket:

```javascript
// Connect to live analytics stream
const ws = new WebSocket('ws://localhost:8000/ws')
ws.onmessage = function (event) {
	const data = JSON.parse(event.data)
	// Process real-time analytics data
}
```

### Custom Models

Replace default models with your own:

1. **YOLO Model**: Place custom YOLO model in project root
2. **Jersey Detection**: Train custom model for jersey number recognition
3. **Segmentation**: Use custom pitch line detection model

### Performance Optimization

For better performance:

-   Use GPU-enabled PyTorch installation
-   Increase `frame_skip_interval` for faster processing
-   Reduce video resolution for real-time analysis
-   Enable hardware acceleration if available

## 📊 Output Data

The system generates comprehensive analytics including:

### Player Data

-   Player ID and tracking confidence
-   Team classification (Team A/Team B)
-   Jersey number (when detectable)
-   Position coordinates (video and pitch space)
-   Movement patterns and speed

### Team Analytics

-   Team formation analysis
-   Player distribution on field
-   Possession statistics
-   Movement heatmaps

### Performance Metrics

-   Processing speed (FPS)
-   Detection accuracy
-   System resource usage
-   Error rates and quality metrics

## 🧪 Testing

Run the test suite to verify functionality:

```bash

# Run specific tests
python -m pytest tests/test_sports_analytics.py -v

# Run all tests
python -m pytest tests/ -v
```

## 🛠️ Development

### Project Structure

The codebase follows clean architecture principles:

-   **Separation of Concerns**: Each module has a specific responsibility
-   **Dependency Injection**: Components are loosely coupled
-   **Testability**: All components can be unit tested
-   **Extensibility**: Easy to add new features and models

### Adding New Features

1. **New Analytics**: Add to `src/analytics/`
2. **New Models**: Add to `src/models/`
3. **New APIs**: Add to `src/api/`
4. **New Utilities**: Add to `src/utils/`

### Code Style

-   Follow PEP 8 Python style guidelines
-   Use type hints for better code documentation
-   Add docstrings to all public methods
-   Write unit tests for new functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `python test_modular_structure.py`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Import Errors**: Ensure you're running from the project root directory

**Memory Issues**: Reduce video resolution or increase frame skip interval

**Model Download Failures**: Check internet connection for YOLO model downloads

**Performance Issues**: Enable GPU support or reduce processing quality

### Getting Help

-   Review configuration: `config/config.yaml`
-   Check logs: `sports_analytics.log`
-   Verify system requirements and dependencies

## 🏆 Features Highlights

-   ✅ **Real-time Processing**: Stream analytics as video plays
-   ✅ **Modular Architecture**: Clean, maintainable codebase
-   ✅ **Web Dashboard**: Interactive browser-based interface
-   ✅ **Multiple Input Formats**: Support for various video formats
-   ✅ **Automated Team Detection**: No manual team labeling required
-   ✅ **Jersey Number Recognition**: OCR-based number detection
-   ✅ **Spatial Mapping**: Video-to-pitch coordinate transformation
-   ✅ **Performance Monitoring**: Built-in performance metrics
-   ✅ **Export Capabilities**: CSV output for further analysis
-   ✅ **Extensible Design**: Easy to add new features and models

---

**Ready to analyze your sports videos? Start with `python main.py` and open http://localhost:8000!** 🚀
