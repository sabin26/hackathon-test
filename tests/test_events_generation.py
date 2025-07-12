#!/usr/bin/env python3
"""
Test that events are being generated properly in simulation.
"""

from dashboard_server import DashboardServer

def test_events_generation():
    """Test that simulation generates events over time"""
    
    print("🔍 Testing Event Generation in Simulation...")
    print("=" * 50)
    
    server = DashboardServer()
    total_events = 0
    event_types = set()
    
    # Generate frames for 10 seconds of simulation
    for i in range(300):  # 300 frames = 10 seconds at 30 FPS
        frame_data = server._generate_realistic_frame_data(i+1, (i+1) * 0.033)
        
        events = frame_data.get("events", [])
        if events:
            total_events += len(events)
            for event in events:
                event_types.add(event.get("type", "unknown"))
                
                # Check event structure
                required_fields = ["type", "timestamp", "player_id", "team", "position", "success"]
                for field in required_fields:
                    if field not in event:
                        print(f"❌ Event missing field: {field}")
                        return False
        
        # Update game stats to accumulate events
        server._update_game_stats(frame_data)
    
    print(f"📊 Total events generated: {total_events}")
    print(f"🎯 Event types: {sorted(event_types)}")
    print(f"📈 Events in game stats: {len(server.game_stats['events'])}")
    
    # Check event analytics
    event_analytics = server.game_stats.get("event_analytics", {})
    event_frequency = event_analytics.get("event_frequency", {})
    
    print("\n📊 Event Frequency:")
    for event_type, count in event_frequency.items():
        if count > 0:
            print(f"   {event_type}: {count}")
    
    # Check success rates
    success_rates = event_analytics.get("event_success_rates", {})
    print("\n📈 Event Success Rates:")
    for event_type, data in success_rates.items():
        if data.get("total", 0) > 0:
            success_rate = data["successful"] / data["total"] * 100
            print(f"   {event_type}: {success_rate:.1f}% ({data['successful']}/{data['total']})")
    
    success = total_events > 0 and len(event_types) > 0
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS: Events are being generated properly!")
        print(f"   Generated {total_events} events of {len(event_types)} different types.")
    else:
        print("❌ FAILURE: No events were generated!")
    
    return success

if __name__ == "__main__":
    test_events_generation()
