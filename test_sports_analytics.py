import unittest
import numpy as np
import tempfile
import os
import yaml
from unittest.mock import Mock, patch
from ai import Config, HomographyManager, TeamIdentifier, SegmentationModel, ObjectTracker

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        config_data = {
            'video': {'path': 'test.mp4'},
            'processing': {'team_n_clusters': 3, 'frame_skip_interval': 2},
            'output': {'csv_path': 'test_output.csv'}
        }
        yaml.dump(config_data, self.temp_config)
        self.temp_config.close()
        
    def tearDown(self):
        os.unlink(self.temp_config.name)
        
    def test_config_validation(self):
        """Test configuration validation"""
        # Test missing video file
        config = Config()
        config.VIDEO_PATH = 'nonexistent_video.mp4'  # Fixed: VIDEO_PATH is a class attribute, not in MODEL_PARAMS
        self.assertFalse(config.validate_config())

        # Test invalid cluster count
        config.MODEL_PARAMS['TEAM_N_CLUSTERS'] = 1
        self.assertFalse(config.validate_config())

class TestHomographyManager(unittest.TestCase):
    def setUp(self):
        self.mock_segmentation = Mock()
        self.homography_manager = HomographyManager(
            pitch_dims=(1050, 680),
            segmentation_model=self.mock_segmentation
        )
        
    def test_line_intersection(self):
        """Test line intersection calculation"""
        line1 = np.array([0, 0, 100, 100])
        line2 = np.array([0, 100, 100, 0])
        
        intersection = self.homography_manager._line_intersection(line1, line2)
        self.assertIsNotNone(intersection)
        self.assertEqual(intersection, [50, 50])
        
    def test_parallel_lines(self):
        """Test parallel line handling"""
        line1 = np.array([0, 0, 100, 0])
        line2 = np.array([0, 50, 100, 50])
        
        intersection = self.homography_manager._line_intersection(line1, line2)
        self.assertIsNone(intersection)
        
    def test_validate_quadrilateral(self):
        """Test quadrilateral validation"""
        # Valid quadrilateral
        valid_corners = [[0, 0], [100, 0], [100, 100], [0, 100]]
        self.assertTrue(self.homography_manager._validate_quadrilateral(valid_corners))
        
        # Invalid quadrilateral (points too close)
        invalid_corners = [[0, 0], [1, 1], [100, 100], [0, 100]]
        self.assertFalse(self.homography_manager._validate_quadrilateral(invalid_corners))

class TestTeamIdentifier(unittest.TestCase):
    def setUp(self):
        self.team_identifier = TeamIdentifier(n_clusters=3)
        
    def test_feature_extraction(self):
        """Test player feature extraction"""
        # Create a test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = [100, 100, 200, 300]  # Valid bounding box
        
        features = self.team_identifier._extract_player_features(frame, bbox)
        self.assertIsNotNone(features)
        if features is not None:
            self.assertEqual(len(features), 16 * 16)  # Should match TEAM_FEATURE_BINS^2
        
    def test_invalid_bbox(self):
        """Test handling of invalid bounding boxes"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        small_bbox = [100, 100, 110, 110]  # Too small
        
        features = self.team_identifier._extract_player_features(frame, small_bbox)
        self.assertIsNone(features)

class TestSegmentationModel(unittest.TestCase):
    def test_placeholder_prediction(self):
        """Test placeholder segmentation prediction"""
        model = SegmentationModel('nonexistent_model.pth')
        
        # Create a test frame with green regions
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:300, 100:500] = [0, 255, 0]  # Green region
        
        mask = model._placeholder_predict(frame)
        self.assertEqual(mask.shape, (480, 640))
        self.assertEqual(mask.dtype, np.uint8)
        
    def test_invalid_frame(self):
        """Test handling of invalid frames"""
        model = SegmentationModel('nonexistent_model.pth')

        # Empty frame
        empty_frame = np.array([])
        mask = model._placeholder_predict(empty_frame)
        self.assertEqual(mask.shape, (0, 0))  # Fixed: Should return (0, 0) shape for 2D array

class TestObjectTracker(unittest.TestCase):
    def setUp(self):
        # Mock YOLO model to avoid downloading
        with patch('ai.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.predict.return_value = []
            mock_model.names = {0: 'person', 32: 'sports ball'}
            mock_yolo.return_value = mock_model
            
            self.tracker = ObjectTracker('yolov8n.pt')
            
    def test_tracking_quality_calculation(self):
        """Test tracking quality calculation"""
        track_id = 1
        
        # Add some tracking history
        self.tracker.tracking_history[track_id] = [
            {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'timestamp': 1.0},
            {'bbox': [105, 105, 205, 205], 'confidence': 0.8, 'timestamp': 2.0},
            {'bbox': [110, 110, 210, 210], 'confidence': 0.85, 'timestamp': 3.0}
        ]
        
        quality = self.tracker._calculate_tracking_quality(track_id)
        self.assertIsInstance(quality, float)
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)

if __name__ == '__main__':
    unittest.main()
