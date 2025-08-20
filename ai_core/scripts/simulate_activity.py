import requests
import json
import random
import time

# The URL where our FastAPI server will be running
API_URL = "http://127.0.0.1:8001/api/v1/ingest-event"

def simulate_user_zero_activity():
    """
    Sends a predefined list of listening events to the ingestion API
    to populate the database for User Zero.
    """
    print("--- 🚀 Starting User Zero Activity Simulation ---")
    
    # A realistic listening history for User Zero (user_id: 1)
    # We assume song IDs from 1 to 30 will exist in our library.
    events = [
        # User listens to a few songs fully
        {'song_id': 5, 'event_type': 'SONG_PLAYED_FULL'},
        {'song_id': 12, 'event_type': 'SONG_PLAYED_FULL'},
        # User skips a song they don't like
        {'song_id': 22, 'event_type': 'SKIP'},
        # Continues listening
        {'song_id': 8, 'event_type': 'SONG_PLAYED_FULL'},
        {'song_id': 15, 'event_type': 'SONG_PLAYED_FULL'},
        {'song_id': 3, 'event_type': 'SONG_PLAYED_FULL'},
        # Skips another
        {'song_id': 19, 'event_type': 'SKIP'},
        # Listens to a few more
        {'song_id': 25, 'event_type': 'SONG_PLAYED_FULL'},
        {'song_id': 2, 'event_type': 'SONG_PLAYED_FULL'},
        {'song_id': 9, 'event_type': 'SONG_PLAYED_FULL'},
    ]
    
    # Add 15 more random events to create a richer history
    for _ in range(15):
        # Weighted choice: user is more likely to listen fully than to skip
        event_type = random.choice(['SONG_PLAYED_FULL', 'SONG_PLAYED_FULL', 'SONG_PLAYED_FULL', 'SKIP'])
        song_id = random.randint(1, 30)
        events.append({'song_id': song_id, 'event_type': event_type})

    headers = {'Content-Type': 'application/json'}
    
    print(f"Preparing to send {len(events)} events to the API...")
    time.sleep(2) # Give a moment to read the log

    for event_data in events:
        # All events are for our "User Zero"
        payload = {
            "user_id": 1,
            "song_id": event_data['song_id'],
            "event_type": event_data['event_type']
        }
        
        try:
            response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
            
            print(f"Sent Event: {payload}")
            if response.status_code == 200:
                print(f"✅ SUCCESS ({response.status_code}): {response.json()}")
            else:
                print(f"❌ FAILED ({response.status_code}): {response.text}")
            print("-" * 20)
            
        except requests.exceptions.ConnectionError as e:
            print(f"\nCRITICAL ERROR: Could not connect to the server at {API_URL}.")
            print("Please ensure the FastAPI server is running before executing this script.")
            break
            
    print("--- Simulation Complete ---")

if __name__ == "__main__":
    simulate_user_zero_activity()
