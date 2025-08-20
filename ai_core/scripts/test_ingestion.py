import requests
import json

# The correct, final URL for the API
API_URL = "http://127.0.0.1:8001/api/v1/ingest-event"

def run_ingestion_test():
    print("--- 🚀 Starting Integration Test for Ingestion Pipeline ---")
    
    test_event = {
        "user_id": 1,
        "song_id": 99,
        "event_type": "SONG_PLAYED_FULL"
    }
    
    print(f"POSTing test event to {API_URL}...")
    try:
        response = requests.post(API_URL, json=test_event)
        if response.status_code == 200:
            print(f"✅ SUCCESS: Server responded with code {response.status_code}.")
            print(f"Response Body: {response.json()}")
        else:
            print(f"❌ FAILED: Server responded with code {response.status_code}.")
            print(f"Response Body: {response.text}")
            
    except Exception as e:
        print(f"❌ FAILED: Could not connect or send event. Error: {e}")

if __name__ == "__main__":
    run_ingestion_test()
