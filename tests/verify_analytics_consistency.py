#!/usr/bin/env python3
"""
Verification script to ensure analytics from video processing and simulation have matching fields.
"""

from typing import Dict, Any, Set
from dashboard_server import DashboardServer

def get_simulation_analytics_structure() -> Dict[str, Any]:
    """Get the analytics structure from simulation mode"""
    server = DashboardServer()

    # Generate multiple frames to ensure all structures are populated
    for i in range(10):
        sample_frame = server._generate_realistic_frame_data(i+1, (i+1) * 0.033)
        server._update_game_stats(sample_frame)

    # Get the last frame for structure analysis
    final_frame = server._generate_realistic_frame_data(11, 11 * 0.033)

    return {
        "frame_data": final_frame,
        "game_stats": server.game_stats
    }

def get_expected_video_processing_structure() -> Dict[str, Any]:
    """Get the expected structure from real video processing based on code analysis"""
    
    # Based on ai.py frame_result structure
    expected_frame_data = {
        "frame_id": "int",
        "timestamp": "float", 
        "objects": [
            {
                "id": "int",
                "type": "str (person/sports ball)",
                "bbox": "list[int]",
                "bbox_video": "list[int]",
                "confidence": "float",
                "pos_pitch": "list[float]",
                "team_name": "str",
                "team": "str", 
                "jersey_number": "str",
                "player_name": "str",
                "player_speed_kmh": "float",
                "distance_covered_m": "float",
                "ball_possession": "bool",
                "confidence_score": "float",
                "detection_count": "int",
                "tracking_quality": "float"
            }
        ],
        "actions": "dict",
        "events": [
            {
                "type": "str",
                "timestamp": "float",
                "player_id": "str/int",
                "team": "str",
                "position": "list[float]",
                "success": "bool"
            }
        ],
        "processing_time": "float",
        "possession_team": "str"
    }
    
    # Expected game stats structure (from dashboard_server.py initialization)
    expected_game_stats = {
        "total_frames": "int",
        "players_detected": "int", 
        "ball_detected": "bool",
        "possession_stats": {"team_A": "int", "team_B": "int", "none": "int"},
        "events": "list",
        "player_positions": "dict",
        "team_colors": "dict",
        "performance_metrics": {
            "fps": "float",
            "processing_time": "float", 
            "detection_accuracy": "float"
        },
        "team_stats": {
            "team_A": {
                "possession_time": "float",
                "total_passes": "int",
                "successful_passes": "int", 
                "shots_taken": "int",
                "shots_on_goal": "int",
                "goals_scored": "int",
                "distance_covered": "float",
                "average_speed": "float",
                "ball_touches": "int",
                "defensive_actions": "int"
            },
            "team_B": "same as team_A"
        },
        "player_stats": {
            "player_id": {
                "team": "str",
                "distance_covered": "float",
                "average_speed": "float",
                "max_speed": "float", 
                "ball_touches": "int",
                "passes_made": "int",
                "passes_received": "int",
                "shots_taken": "int",
                "possession_time": "float",
                "last_position": "dict/None",
                "speed_samples": "list"
            }
        },
        "game_flow": {
            "possession_changes": "list",
            "momentum_indicator": "float",
            "activity_zones": "dict",
            "game_intensity": "float"
        },
        "event_analytics": {
            "event_frequency": "dict",
            "event_success_rates": "dict", 
            "event_timeline": "list",
            "heat_zones": "dict"
        }
    }
    
    return {
        "frame_data": expected_frame_data,
        "game_stats": expected_game_stats
    }

def extract_field_names(obj, prefix="") -> Set[str]:
    """Recursively extract all field names from a nested structure"""
    fields = set()
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            field_name = f"{prefix}.{key}" if prefix else key
            fields.add(field_name)
            
            if isinstance(value, (dict, list)) and value:
                if isinstance(value, list) and len(value) > 0:
                    # For lists, check the first item
                    fields.update(extract_field_names(value[0], field_name))
                else:
                    fields.update(extract_field_names(value, field_name))
    
    elif isinstance(obj, list) and len(obj) > 0:
        # For lists, check the first item
        fields.update(extract_field_names(obj[0], prefix))
    
    return fields

def compare_analytics_structures():
    """Compare simulation and expected video processing analytics structures"""
    
    print("🔍 Analyzing analytics structure consistency...")
    print("=" * 60)
    
    # Get structures
    simulation_data = get_simulation_analytics_structure()
    expected_data = get_expected_video_processing_structure()
    
    # Extract field names
    sim_frame_fields = extract_field_names(simulation_data["frame_data"])
    exp_frame_fields = extract_field_names(expected_data["frame_data"])
    
    sim_stats_fields = extract_field_names(simulation_data["game_stats"])
    exp_stats_fields = extract_field_names(expected_data["game_stats"])
    
    print("📊 FRAME DATA COMPARISON:")
    print("-" * 30)
    
    # Check frame data fields
    missing_in_sim = exp_frame_fields - sim_frame_fields
    extra_in_sim = sim_frame_fields - exp_frame_fields
    
    if missing_in_sim:
        print("❌ Missing in simulation:")
        for field in sorted(missing_in_sim):
            print(f"   - {field}")
    
    if extra_in_sim:
        print("➕ Extra in simulation:")
        for field in sorted(extra_in_sim):
            print(f"   - {field}")
    
    if not missing_in_sim and not extra_in_sim:
        print("✅ Frame data structures match perfectly!")
    
    print("\n📈 GAME STATS COMPARISON:")
    print("-" * 30)
    
    # Check game stats fields  
    missing_stats = exp_stats_fields - sim_stats_fields
    extra_stats = sim_stats_fields - exp_stats_fields
    
    if missing_stats:
        print("❌ Missing in simulation:")
        for field in sorted(missing_stats):
            print(f"   - {field}")
    
    if extra_stats:
        print("➕ Extra in simulation:")
        for field in sorted(extra_stats):
            print(f"   - {field}")
    
    if not missing_stats and not extra_stats:
        print("✅ Game stats structures match perfectly!")
    
    print("\n" + "=" * 60)
    
    # Overall summary
    total_issues = len(missing_in_sim) + len(extra_in_sim) + len(missing_stats) + len(extra_stats)
    
    if total_issues == 0:
        print("🎉 SUCCESS: All analytics structures are consistent!")
        print("   Simulation and video processing will produce matching analytics.")
    else:
        print(f"⚠️  ISSUES FOUND: {total_issues} field mismatches detected.")
        print("   Please review and fix the inconsistencies above.")
    
    return total_issues == 0

if __name__ == "__main__":
    compare_analytics_structures()
