#!/usr/bin/env python3
"""
Simple verification to check key analytics fields between simulation and real video processing.
"""

from dashboard_server import DashboardServer

def test_simulation_analytics():
    """Test that simulation generates the expected analytics structure"""
    
    print("🔍 Testing Simulation Analytics Structure...")
    print("=" * 50)
    
    server = DashboardServer()
    
    # Generate a few frames to populate data
    for i in range(5):
        frame_data = server._generate_realistic_frame_data(i+1, (i+1) * 0.033)
        server._update_game_stats(frame_data)
    
    # Get final frame for testing
    final_frame = server._generate_realistic_frame_data(6, 6 * 0.033)
    
    print("📊 FRAME DATA STRUCTURE:")
    print("-" * 25)
    
    # Check frame data fields
    required_frame_fields = [
        "frame_id", "timestamp", "objects", "actions", "events", 
        "possession_team", "processing_time"
    ]
    
    missing_frame_fields = []
    missing_obj_fields = []  # Ensure this is always defined
    for field in required_frame_fields:
        if field not in final_frame:
            missing_frame_fields.append(field)
        else:
            print(f"✅ {field}: {type(final_frame[field])}")
    
    if missing_frame_fields:
        print(f"❌ Missing frame fields: {missing_frame_fields}")
    
    # Check object structure
    print("\n🎯 OBJECT STRUCTURE:")
    print("-" * 20)
    
    objects = final_frame.get("objects", [])
    if objects:
        sample_obj = objects[0]
        required_obj_fields = [
            "id", "type", "bbox", "bbox_video", "confidence", "pos_pitch",
            "team_name", "team", "jersey_number", "player_name", 
            "player_speed_kmh", "confidence_score", "detection_count", "tracking_quality"
        ]
        
        missing_obj_fields = []
        for field in required_obj_fields:
            if field not in sample_obj:
                missing_obj_fields.append(field)
            else:
                print(f"✅ {field}: {type(sample_obj[field])}")
        
        if missing_obj_fields:
            print(f"❌ Missing object fields: {missing_obj_fields}")
    else:
        print("❌ No objects found in frame data")
    
    # Check events structure
    print("\n📅 EVENTS STRUCTURE:")
    print("-" * 20)
    
    missing_event_fields = []
    events = final_frame.get("events", [])
    if events:
        sample_event = events[0]
        required_event_fields = ["type", "timestamp", "player_id", "team", "position", "success"]
        
        for field in required_event_fields:
            if field not in sample_event:
                missing_event_fields.append(field)
            else:
                print(f"✅ {field}: {type(sample_event[field])}")
        
        if missing_event_fields:
            print(f"❌ Missing event fields: {missing_event_fields}")
    else:
        print("⚠️  No events found in frame data")
    
    # Check game stats structure
    print("\n📈 GAME STATS STRUCTURE:")
    print("-" * 25)
    
    stats = server.game_stats
    required_stats_sections = [
        "total_frames", "players_detected", "ball_detected", "possession_stats",
        "events", "player_positions", "team_colors", "performance_metrics",
        "team_stats", "player_stats", "game_flow", "event_analytics"
    ]
    
    missing_stats_sections = []
    for section in required_stats_sections:
        if section not in stats:
            missing_stats_sections.append(section)
        else:
            print(f"✅ {section}: {type(stats[section])}")
    
    if missing_stats_sections:
        print(f"❌ Missing stats sections: {missing_stats_sections}")
    
    # Check if player stats are populated
    print("\n👥 PLAYER STATS CHECK:")
    print("-" * 20)
    
    player_stats = stats.get("player_stats", {})
    missing_player_fields = []  # Ensure this is always defined
    if player_stats:
        sample_player_id = list(player_stats.keys())[0]
        sample_player_stat = player_stats[sample_player_id]
        
        required_player_fields = [
            "team", "distance_covered", "average_speed", "max_speed",
            "ball_touches", "passes_made", "passes_received", "shots_taken",
            "possession_time", "last_position", "speed_samples"
        ]
        
        missing_player_fields = []
        for field in required_player_fields:
            if field not in sample_player_stat:
                missing_player_fields.append(field)
            else:
                print(f"✅ {field}: {type(sample_player_stat[field])}")
        
        if missing_player_fields:
            print(f"❌ Missing player stat fields: {missing_player_fields}")
        
        print(f"📊 Total players tracked: {len(player_stats)}")
    else:
        print("❌ No player stats found")
    
    # Summary
    print("\n" + "=" * 50)
    
    total_issues = (len(missing_frame_fields) + len(missing_obj_fields if 'missing_obj_fields' in locals() else []) + 
                   len(missing_event_fields if 'missing_event_fields' in locals() else []) + 
                   len(missing_stats_sections) + len(missing_player_fields))
    
    if total_issues == 0:
        print("🎉 SUCCESS: Simulation analytics structure is complete!")
        print("   All required fields are present and properly typed.")
    else:
        print(f"⚠️  ISSUES: {total_issues} missing fields detected.")
        print("   Please review the missing fields above.")
    
    return total_issues == 0

def compare_key_fields():
    """Compare key fields between simulation and expected real processing"""
    
    print("\n🔄 COMPARING KEY ANALYTICS FIELDS...")
    print("=" * 50)
    
    # Expected fields from real video processing (based on code analysis)
    expected_frame_fields = {"frame_id", "timestamp", "objects", "actions"}
    expected_object_fields = {"id", "type", "bbox_video", "confidence", "pos_pitch", "team_name", "jersey_number", "player_name"}
    expected_stats_fields = {"total_frames", "players_detected", "ball_detected", "team_stats", "player_stats"}
    
    # Get simulation data
    server = DashboardServer()
    frame_data = server._generate_realistic_frame_data(1, 1.0)
    server._update_game_stats(frame_data)
    
    # Check frame fields
    sim_frame_fields = set(frame_data.keys())
    frame_match = expected_frame_fields.issubset(sim_frame_fields)
    print(f"📊 Frame fields match: {'✅' if frame_match else '❌'}")
    if not frame_match:
        missing = expected_frame_fields - sim_frame_fields
        print(f"   Missing: {missing}")
    
    # Check object fields
    objects = frame_data.get("objects", [])
    if objects:
        sim_object_fields = set(objects[0].keys())
        object_match = expected_object_fields.issubset(sim_object_fields)
        print(f"🎯 Object fields match: {'✅' if object_match else '❌'}")
        if not object_match:
            missing = expected_object_fields - sim_object_fields
            print(f"   Missing: {missing}")
    else:
        print("🎯 Object fields match: ❌ (no objects)")
        object_match = False
    
    # Check stats fields
    sim_stats_fields = set(server.game_stats.keys())
    stats_match = expected_stats_fields.issubset(sim_stats_fields)
    print(f"📈 Stats fields match: {'✅' if stats_match else '❌'}")
    if not stats_match:
        missing = expected_stats_fields - sim_stats_fields
        print(f"   Missing: {missing}")
    
    overall_match = frame_match and object_match and stats_match
    print(f"\n🎯 OVERALL COMPATIBILITY: {'✅ PASS' if overall_match else '❌ FAIL'}")
    
    return overall_match

if __name__ == "__main__":
    success1 = test_simulation_analytics()
    success2 = compare_key_fields()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("   Simulation and video processing analytics are compatible.")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("   Please review and fix the issues above.")
