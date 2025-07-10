"""
Real-time Sports Analytics Dashboard Server

This module provides a FastAPI-based WebSocket server for streaming
real-time sports analytics data to a web dashboard.
"""

import asyncio
import json
import logging
import time
import queue
import os
import shutil
from typing import Dict, List, Any
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardServer:
    """WebSocket server for real-time sports analytics dashboard"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI(title="Sports Analytics Dashboard")
        self.active_connections: List[WebSocket] = []
        self.latest_data: Dict[str, Any] = {}
        self.data_queue = queue.Queue(maxsize=100)  # Queue for thread-safe data passing
        self.game_stats: Dict[str, Any] = {
            "total_frames": 0,
            "players_detected": 0,
            "ball_detected": False,
            "possession_stats": {"team_A": 0, "team_B": 0, "none": 0},
            "events": [],
            "player_positions": {},
            "team_colors": {},
            "performance_metrics": {
                "fps": 0,
                "processing_time": 0,
                "detection_accuracy": 0
            }
        }

        # Video processing state
        self.current_video_path = None
        self.video_processor = None
        self.processing_thread = None
        self.is_processing = False

        # Create uploads directory
        self.uploads_dir = Path("uploads")
        self.uploads_dir.mkdir(exist_ok=True)

        self._setup_routes()
        self._setup_static_files()

        # Start background task to process queue
        self._queue_processor_task = None
    
    def _setup_static_files(self):
        """Setup static file serving for dashboard assets"""
        # Create static directory if it doesn't exist
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)
        
        # Create templates directory if it doesn't exist
        templates_dir = Path("templates")
        templates_dir.mkdir(exist_ok=True)
        
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        self.templates = Jinja2Templates(directory="templates")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Serve the main dashboard page"""
            return self.templates.TemplateResponse(
                "dashboard.html", 
                {"request": request, "title": "Sports Analytics Dashboard"}
            )
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time data streaming"""
            await self.connect(websocket)
            try:
                while True:
                    # Keep connection alive and handle any incoming messages
                    data = await websocket.receive_text()
                    # Handle client messages if needed (e.g., control commands)
                    await self.handle_client_message(websocket, data)
            except WebSocketDisconnect:
                self.disconnect(websocket)
        
        @self.app.get("/api/stats")
        async def get_stats():
            """Get current game statistics"""
            return self.game_stats
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "connections": len(self.active_connections)}

        @self.app.post("/api/upload-video")
        async def upload_video(file: UploadFile = File(...)):
            """Handle video file upload"""
            try:
                # Validate file type
                allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv']
                file_extension = Path(file.filename).suffix.lower()

                if file_extension not in allowed_extensions:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
                    )

                # Save uploaded file
                file_path = self.uploads_dir / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                self.current_video_path = str(file_path)

                logger.info(f"Video uploaded successfully: {file.filename}")
                return JSONResponse({
                    "status": "success",
                    "message": f"Video '{file.filename}' uploaded successfully",
                    "filename": file.filename,
                    "path": str(file_path)
                })

            except Exception as e:
                logger.error(f"Video upload failed: {e}")
                raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

        @self.app.post("/api/start-analysis")
        async def start_analysis():
            """Start video analysis"""
            try:
                if not self.current_video_path or not os.path.exists(self.current_video_path):
                    raise HTTPException(status_code=400, detail="No video file uploaded")

                if self.is_processing:
                    raise HTTPException(status_code=400, detail="Analysis already in progress")

                # Start video processing in background
                await self._start_video_processing()

                return JSONResponse({
                    "status": "success",
                    "message": "Video analysis started"
                })

            except Exception as e:
                logger.error(f"Failed to start analysis: {e}")
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

        @self.app.post("/api/stop-analysis")
        async def stop_analysis():
            """Stop video analysis"""
            try:
                await self._stop_video_processing()

                return JSONResponse({
                    "status": "success",
                    "message": "Video analysis stopped"
                })

            except Exception as e:
                logger.error(f"Failed to stop analysis: {e}")
                raise HTTPException(status_code=500, detail=f"Stop failed: {str(e)}")

        @self.app.get("/api/status")
        async def get_processing_status():
            """Get current processing status"""
            return JSONResponse({
                "is_processing": self.is_processing,
                "current_video": Path(self.current_video_path).name if self.current_video_path else None,
                "connections": len(self.active_connections)
            })
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
        
        # Send initial data to new client
        if self.latest_data:
            await websocket.send_text(json.dumps({
                "type": "initial_data",
                "data": self.latest_data,
                "stats": self.game_stats
            }))
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def handle_client_message(self, websocket: WebSocket, message: str):
        """Handle messages from clients"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message_type == "request_stats":
                await websocket.send_text(json.dumps({
                    "type": "stats_update",
                    "stats": self.game_stats
                }))
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received from client: {message}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def broadcast_frame_data(self, frame_data: Dict[str, Any]):
        """Broadcast new frame data to all connected clients"""
        if not self.active_connections:
            return
        
        # Update latest data
        self.latest_data = frame_data
        
        # Update game statistics
        self._update_game_stats(frame_data)
        
        # Prepare message for clients
        message = {
            "type": "frame_update",
            "data": frame_data,
            "stats": self.game_stats,
            "timestamp": time.time()
        }
        
        # Broadcast to all connected clients
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send data to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    def _update_game_stats(self, frame_data: Dict[str, Any]):
        """Update game statistics based on new frame data"""
        try:
            self.game_stats["total_frames"] += 1
            
            # Update player and ball detection stats
            objects = frame_data.get("objects", [])
            players = [obj for obj in objects if obj.get("type") == "person"]
            balls = [obj for obj in objects if obj.get("type") == "sports ball"]
            
            self.game_stats["players_detected"] = len(players)
            self.game_stats["ball_detected"] = len(balls) > 0
            
            # Update player positions for heatmap
            for player in players:
                player_id = player.get("id")
                pos_pitch = player.get("pos_pitch", [0, 0])
                team_name = player.get("team_name", "unknown")
                
                if player_id:
                    if player_id not in self.game_stats["player_positions"]:
                        self.game_stats["player_positions"][player_id] = []
                    
                    self.game_stats["player_positions"][player_id].append({
                        "x": pos_pitch[0] if len(pos_pitch) > 0 else 0,
                        "y": pos_pitch[1] if len(pos_pitch) > 1 else 0,
                        "timestamp": frame_data.get("timestamp", 0),
                        "team": team_name
                    })
                    
                    # Keep only recent positions (last 100 points)
                    if len(self.game_stats["player_positions"][player_id]) > 100:
                        self.game_stats["player_positions"][player_id] = \
                            self.game_stats["player_positions"][player_id][-100:]
            
            # Update possession stats
            actions = frame_data.get("actions", {})
            for player_id, action_data in actions.items():
                event_type = action_data.get("event_type", "")
                if event_type == "Possession":
                    team_name = action_data.get("team_name", "none")
                    if team_name in self.game_stats["possession_stats"]:
                        self.game_stats["possession_stats"][team_name] += 1
                    else:
                        self.game_stats["possession_stats"]["none"] += 1
            
            # Update events list
            for player_id, action_data in actions.items():
                event_type = action_data.get("event_type", "")
                if event_type and event_type != "":
                    event = {
                        "timestamp": frame_data.get("timestamp", 0),
                        "player_id": player_id,
                        "event_type": event_type,
                        "event_outcome": action_data.get("event_outcome", ""),
                        "team_name": action_data.get("team_name", "unknown")
                    }
                    self.game_stats["events"].append(event)
                    
                    # Keep only recent events (last 50)
                    if len(self.game_stats["events"]) > 50:
                        self.game_stats["events"] = self.game_stats["events"][-50:]
        
        except Exception as e:
            logger.error(f"Error updating game stats: {e}")
    
    async def _process_data_queue(self):
        """Background task to process data queue and broadcast to clients"""
        while True:
            try:
                # Check for new data in queue (non-blocking)
                try:
                    frame_data = self.data_queue.get_nowait()
                    await self.broadcast_frame_data(frame_data)
                    self.data_queue.task_done()
                except queue.Empty:
                    # No data available, wait a bit
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(0.1)

    async def start_server(self):
        """Start the dashboard server"""
        logger.info(f"Starting dashboard server on {self.host}:{self.port}")

        # Start queue processor
        self._queue_processor_task = asyncio.create_task(self._process_data_queue())

        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        try:
            await server.serve()
        finally:
            # Cancel queue processor when server stops
            if self._queue_processor_task:
                self._queue_processor_task.cancel()
                try:
                    await self._queue_processor_task
                except asyncio.CancelledError:
                    pass
    
    async def _start_video_processing(self):
        """Start video processing in background thread"""
        try:
            from ai import Config, VideoProcessor
            import threading

            # Load configuration
            config = Config()
            config.load_from_yaml("config.yaml")

            # Override video path with uploaded file
            config.VIDEO_PATH = self.current_video_path

            # Validate configuration
            if not config.validate_config():
                raise Exception("Configuration validation failed")

            # Create video processor with streaming enabled
            self.video_processor = VideoProcessor(config, enable_streaming=True)

            # Set the video path for runtime upload
            if self.current_video_path:
                self.video_processor.set_video_path(self.current_video_path)

            # Set the dashboard server for streaming
            if hasattr(self.video_processor, 'dashboard_server'):
                self.video_processor.dashboard_server = self

            # Start processing in background thread
            def process_video():
                try:
                    self.is_processing = True
                    logger.info(f"Starting video analysis: {self.current_video_path}")
                    self.video_processor.start()
                    logger.info("Video analysis completed")
                except Exception as e:
                    logger.error(f"Video processing failed: {e}")
                finally:
                    self.is_processing = False

            self.processing_thread = threading.Thread(target=process_video, daemon=True)
            self.processing_thread.start()

        except Exception as e:
            self.is_processing = False
            logger.error(f"Failed to start video processing: {e}")
            raise

    async def _stop_video_processing(self):
        """Stop video processing"""
        try:
            self.is_processing = False

            if self.video_processor and hasattr(self.video_processor, 'stop_event'):
                self.video_processor.stop_event.set()

            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)

            logger.info("Video processing stopped")

        except Exception as e:
            logger.error(f"Failed to stop video processing: {e}")
            raise

    def run(self):
        """Run the server (blocking)"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )

# Global server instance for external access
dashboard_server = DashboardServer()

if __name__ == "__main__":
    dashboard_server.run()
