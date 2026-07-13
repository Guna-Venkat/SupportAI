import os
import subprocess
import time
import requests
import json
import sys
from pathlib import Path

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)
Path("scratch").mkdir(exist_ok=True)

# Clear old traces.jsonl
traces_file = Path("logs/traces.jsonl")
if traces_file.exists():
    try:
        traces_file.unlink()
    except Exception as e:
        print(f"Could not delete old traces file: {e}")

print("Launching FastAPI server in background...")
env = os.environ.copy()
env["TESTING"] = "true"  # Use fast test backend
env["PYTHONUNBUFFERED"] = "1"  # Do not buffer output logs

# Direct server log
server_log = open("scratch/uvicorn.log", "w", encoding="utf-8")

# Run uvicorn as a subprocess
server_process = subprocess.Popen(
    ["uvicorn", "src.api.app:app", "--host", "127.0.0.1", "--port", "8000"],
    env=env,
    stdout=server_log,
    stderr=server_log
)

# Wait for server to start (30 seconds to be safe on CPU)
print("Waiting for server to initialize (30s)...")
time.sleep(30)

# Check if process is still running
poll = server_process.poll()
if poll is not None:
    print(f"ERROR: Server process exited early with code {poll}!")
    server_log.close()
    with open("scratch/uvicorn.log", "r", encoding="utf-8") as f:
        print("--- Uvicorn Log Output ---")
        print(f.read())
    sys.exit(1)

base_url = "http://127.0.0.1:8000"

try:
    # 1. /health
    print("\n--- Testing GET /health ---")
    r = requests.get(f"{base_url}/health")
    print(f"Status: {r.status_code}")
    print(f"Headers X-Request-ID: {r.headers.get('X-Request-ID')}")
    print(f"Response: {r.json()}")

    # 2. /version
    print("\n--- Testing GET /version ---")
    r = requests.get(f"{base_url}/version")
    print(f"Status: {r.status_code}")
    print(f"Headers X-Request-ID: {r.headers.get('X-Request-ID')}")
    print(f"Response: {r.json()}")

    # 3. /predict (High Confidence -> auto-route)
    print("\n--- Testing POST /predict (High Confidence) ---")
    payload = {"text": "I forgot my passcode and cannot login"}
    r = requests.post(f"{base_url}/predict", json=payload)
    print(f"Status: {r.status_code}")
    print(f"Headers X-Request-ID: {r.headers.get('X-Request-ID')}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")

    # 4. /predict (Mid Confidence -> LLM Fallback)
    print("\n--- Testing POST /predict (Mid Confidence/Fallback) ---")
    # Using text that gives medium confidence
    payload = {"text": "reset card pin number"}
    r = requests.post(f"{base_url}/predict", json=payload)
    print(f"Status: {r.status_code}")
    print(f"Headers X-Request-ID: {r.headers.get('X-Request-ID')}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")

    # 5. /retrieve
    print("\n--- Testing POST /retrieve ---")
    payload = {"query": "reset pin number", "top_k": 2}
    r = requests.post(f"{base_url}/retrieve", json=payload)
    print(f"Status: {r.status_code}")
    print(f"Headers X-Request-ID: {r.headers.get('X-Request-ID')}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")

    # 6. /explain
    print("\n--- Testing POST /explain ---")
    payload = {"text": "I forgot my passcode and cannot login", "num_features": 5, "num_samples": 20}
    r = requests.post(f"{base_url}/explain", json=payload)
    print(f"Status: {r.status_code}")
    print(f"Headers X-Request-ID: {r.headers.get('X-Request-ID')}")
    # Print only first 300 chars of HTML explanation to keep log readable
    resp_data = r.json()
    resp_data["explanation_html"] = resp_data["explanation_html"][:100] + "..."
    print(f"Response (truncated HTML): {json.dumps(resp_data, indent=2)}")

    # 7. /metrics
    print("\n--- Testing GET /metrics ---")
    r = requests.get(f"{base_url}/metrics")
    print(f"Status: {r.status_code}")
    # Print first few lines of metrics
    print("\n".join(r.text.splitlines()[:15]))

    # Verify structured logging (traces.jsonl)
    print("\n--- Checking logs/traces.jsonl ---")
    time.sleep(2)  # Wait for file write to complete
    if traces_file.exists():
        with open(traces_file, "r", encoding="utf-8") as f:
            for line in f:
                print(line.strip())
    else:
        print("traces.jsonl does not exist!")

finally:
    print("\nTerminating FastAPI server...")
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()
    server_log.close()
    print("FastAPI server terminated.")
