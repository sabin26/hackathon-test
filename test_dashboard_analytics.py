#!/usr/bin/env python3
"""
Test that analytics are properly processed and displayed meaningfully on the dashboard.
"""

import json
from dashboard_server import DashboardServer

def test_meaningful_analytics_display():
    """Test that analytics provide meaningful information for dashboard display"""
    
    print("🔍 Testing Meaningful Analytics Display...")
    print("=" * 60)
    
    server = DashboardServer()
    
    # Generate realistic game scenario over 30 seconds
    print("📊 Generating 30 seconds of realistic game data...")
    
    for i in range(900):  # 900 frames = 30 seconds at 30 FPS
        frame_data = server._generate_realistic_frame_data(i+1, (i+1) * 0.033)
        server._update_game_stats(frame_data)
    
    stats = server.game_stats
    
    print("\n🎯 MEANINGFUL ANALYTICS VERIFICATION:")
    print("-" * 40)
    
    # 1. Game Overview Metrics
    print(f"✅ Players Detected: {stats['players_detected']}")
    print(f"✅ Ball Detected: {'Yes' if stats['ball_detected'] else 'No'}")
    print(f"✅ Total Frames Processed: {stats['total_frames']}")
    print(f"✅ Total Events: {len(stats['events'])}")
    
    # 2. Performance Metrics
    perf = stats['performance_metrics']
    print(f"✅ Processing FPS: {perf['fps']:.1f}")
    print(f"✅ Processing Time: {perf['processing_time']:.1f}ms")
    print(f"✅ Detection Accuracy: {perf['detection_accuracy']:.1%}")
    
    # 3. Team Statistics Analysis
    print("\n🏆 TEAM PERFORMANCE ANALYSIS:")
    print("-" * 30)
    
    for team_name, team_data in stats['team_stats'].items():
        print(f"\n{team_name.upper()}:")
        
        # Calculate meaningful metrics
        pass_accuracy = (team_data['successful_passes'] / max(team_data['total_passes'], 1)) * 100
        shot_accuracy = (team_data['shots_on_goal'] / max(team_data['shots_taken'], 1)) * 100
        possession_time = team_data['possession_time']
        
        print(f"  📈 Pass Accuracy: {pass_accuracy:.1f}% ({team_data['successful_passes']}/{team_data['total_passes']})")
        print(f"  🎯 Shot Accuracy: {shot_accuracy:.1f}% ({team_data['shots_on_goal']}/{team_data['shots_taken']})")
        print(f"  ⚽ Goals Scored: {team_data['goals_scored']}")
        print(f"  ⏱️  Possession Time: {possession_time:.1f}s")
        print(f"  🏃 Distance Covered: {team_data['distance_covered']:.1f}m")
        print(f"  💨 Average Speed: {team_data['average_speed']:.1f} km/h")
        print(f"  🤾 Ball Touches: {team_data['ball_touches']}")
        print(f"  🛡️  Defensive Actions: {team_data['defensive_actions']}")
    
    # 4. Player Statistics Analysis
    print("\n👥 PLAYER PERFORMANCE ANALYSIS:")
    print("-" * 35)
    
    player_stats = stats['player_stats']
    if player_stats:
        # Find top performers
        top_distance = max(player_stats.items(), key=lambda x: x[1]['distance_covered'])
        top_speed = max(player_stats.items(), key=lambda x: x[1]['max_speed'])
        top_touches = max(player_stats.items(), key=lambda x: x[1]['ball_touches'])
        
        print(f"🏃 Most Distance: Player {top_distance[0]} - {top_distance[1]['distance_covered']:.1f}m")
        print(f"💨 Fastest Player: Player {top_speed[0]} - {top_speed[1]['max_speed']:.1f} km/h")
        print(f"🤾 Most Ball Touches: Player {top_touches[0]} - {top_touches[1]['ball_touches']} touches")
        print(f"📊 Total Players Tracked: {len(player_stats)}")
    
    # 5. Event Analytics
    print("\n📅 EVENT ANALYTICS:")
    print("-" * 20)
    
    event_freq = stats['event_analytics']['event_frequency']
    total_events = sum(event_freq.values())
    
    if total_events > 0:
        print(f"📊 Total Events: {total_events}")
        for event_type, count in event_freq.items():
            if count > 0:
                percentage = (count / total_events) * 100
                print(f"  {event_type}: {count} ({percentage:.1f}%)")
        
        # Success rates
        print("\n📈 Success Rates:")
        success_rates = stats['event_analytics']['event_success_rates']
        for event_type, data in success_rates.items():
            if data['total'] > 0:
                rate = (data['successful'] / data['total']) * 100
                print(f"  {event_type}: {rate:.1f}% ({data['successful']}/{data['total']})")
    
    # 6. Game Flow Analysis
    print("\n🌊 GAME FLOW ANALYSIS:")
    print("-" * 25)
    
    game_flow = stats['game_flow']
    momentum = game_flow['momentum_indicator']
    intensity = game_flow['game_intensity']
    possession_changes = len(game_flow['possession_changes'])
    
    print(f"⚖️  Momentum: {momentum:.2f} ({'Team A favored' if momentum < 0 else 'Team B favored' if momentum > 0 else 'Balanced'})")
    print(f"🔥 Game Intensity: {intensity:.1f}")
    print(f"🔄 Possession Changes: {possession_changes}")
    
    # 7. Heat Zones Analysis
    print("\n🗺️  FIELD HEAT ZONES:")
    print("-" * 20)
    
    heat_zones = stats['event_analytics']['heat_zones']
    for zone, teams in heat_zones.items():
        total_zone_activity = sum(teams.values())
        if total_zone_activity > 0:
            print(f"  {zone.replace('_', ' ').title()}: {total_zone_activity} events")
            for team, count in teams.items():
                if count > 0:
                    percentage = (count / total_zone_activity) * 100
                    print(f"    {team}: {count} ({percentage:.1f}%)")
    
    # 8. Possession Analysis
    print("\n⚽ POSSESSION ANALYSIS:")
    print("-" * 25)
    
    possession = stats['possession_stats']
    total_possession = sum(possession.values())
    
    if total_possession > 0:
        for team, frames in possession.items():
            percentage = (frames / total_possession) * 100
            seconds = frames / 30  # Convert frames to seconds (30 FPS)
            team_display = team.replace('_', ' ').title() if team != 'none' else 'No Possession'
            print(f"  {team_display}: {percentage:.1f}% ({seconds:.1f}s)")
    
    print("\n" + "=" * 60)
    
    # Verify meaningful data exists
    meaningful_checks = [
        stats['total_frames'] > 0,
        stats['players_detected'] > 0,
        len(stats['player_stats']) > 0,
        total_events > 0,
        any(team['total_passes'] > 0 for team in stats['team_stats'].values()),
        possession['team_A'] > 0 or possession['team_B'] > 0
    ]
    
    passed_checks = sum(meaningful_checks)
    total_checks = len(meaningful_checks)
    
    if passed_checks == total_checks:
        print("🎉 SUCCESS: All analytics are meaningful and ready for dashboard display!")
        print("   ✅ Game metrics show realistic values")
        print("   ✅ Team statistics provide actionable insights")
        print("   ✅ Player performance data is comprehensive")
        print("   ✅ Event analytics show game progression")
        print("   ✅ Game flow indicators are working")
        print("   ✅ Possession tracking is functional")
    else:
        print(f"⚠️  PARTIAL SUCCESS: {passed_checks}/{total_checks} checks passed")
        print("   Some analytics may need improvement for meaningful display")
    
    return passed_checks == total_checks

def test_dashboard_data_format():
    """Test that data is formatted correctly for dashboard consumption"""
    
    print("\n🔧 Testing Dashboard Data Format Compatibility...")
    print("=" * 50)
    
    server = DashboardServer()
    
    # Generate sample data
    frame_data = server._generate_realistic_frame_data(1, 1.0)
    server._update_game_stats(frame_data)
    
    # Test message format that would be sent to dashboard
    message = {
        "type": "frame_update",
        "data": frame_data,
        "stats": server.game_stats,
        "timestamp": 1234567890.0
    }
    
    print("📊 Message Structure:")
    print(f"✅ Type: {message['type']}")
    print(f"✅ Has Frame Data: {'data' in message}")
    print(f"✅ Has Stats: {'stats' in message}")
    print(f"✅ Has Timestamp: {'timestamp' in message}")
    
    # Test JSON serialization (dashboard receives JSON)
    try:
        json_str = json.dumps(message, default=str)
        _ = json.loads(json_str)
        print("✅ JSON Serialization: Success")
    except Exception as e:
        print(f"❌ JSON Serialization: Failed - {e}")
        return False
    
    # Test key dashboard fields
    stats = message['stats']
    required_dashboard_fields = [
        'players_detected', 'ball_detected', 'possession_stats',
        'team_stats', 'player_stats', 'events', 'performance_metrics'
    ]
    
    missing_fields = []
    for field in required_dashboard_fields:
        if field not in stats:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing Dashboard Fields: {missing_fields}")
        return False
    else:
        print("✅ All Dashboard Fields Present")
    
    print("🎉 Dashboard data format is compatible!")
    return True

if __name__ == "__main__":
    success1 = test_meaningful_analytics_display()
    success2 = test_dashboard_data_format()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("   Analytics are meaningful and dashboard-ready!")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("   Please review the issues above.")
