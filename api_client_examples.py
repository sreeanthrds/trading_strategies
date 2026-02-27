"""
FastAPI Client Examples
=======================
Examples of how to use the 5EMA Strategy Backtest API.
"""

import requests
import json
import time
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test API health."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    return response.json()

def run_single_day_async():
    """Run single day backtest asynchronously."""
    payload = {
        "date": "2024-10-03",
        "base_target": 25.0,
        "signal_start_time": "09:20",
        "daily_target": 100.0,
        "run_async": True,
        "save_results": True
    }
    
    response = requests.post(f"{BASE_URL}/backtest", json=payload)
    print("Started backtest:", response.json())
    
    task_id = response.json()["task_id"]
    
    # Poll for completion
    while True:
        status = requests.get(f"{BASE_URL}/backtest/{task_id}/status").json()
        print(f"Status: {status['status']}")
        
        if status['status'] in ['completed', 'failed']:
            break
        
        time.sleep(2)
    
    # Get result
    result = requests.get(f"{BASE_URL}/backtest/{task_id}/result").json()
    print("Result:", json.dumps(result, indent=2))
    
    return result

def run_date_range_sync():
    """Run date range backtest synchronously."""
    payload = {
        "start_date": "2024-10-01",
        "end_date": "2024-10-05",
        "base_target": 20.0,
        "allow_opposite_entry": True,
        "historical_candles": 300,
        "run_async": False,
        "save_results": True
    }
    
    response = requests.post(f"{BASE_URL}/backtest", json=payload)
    print("Backtest completed:", response.json())
    
    return response.json()

def run_scalping_config():
    """Run with scalping configuration."""
    payload = {
        "date": "2024-10-03",
        "base_target": 10.0,
        "signal_start_time": "09:17",
        "force_exit_time": "15:25",
        "allow_opposite_entry": True,
        "daily_target": 100.0,
        "historical_candles": 200,
        "run_async": False
    }
    
    response = requests.post(f"{BASE_URL}/backtest", json=payload)
    print("Scalping config result:", response.json())
    
    return response.json()

def list_all_tasks():
    """List all tasks."""
    response = requests.get(f"{BASE_URL}/backtest/tasks")
    print("All tasks:", json.dumps(response.json(), indent=2))
    return response.json()

def download_results(task_id):
    """Download backtest results."""
    response = requests.get(f"{BASE_URL}/backtest/{task_id}/download")
    
    if response.status_code == 200:
        filename = f"results_{task_id}.csv"
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Results downloaded to {filename}")
    else:
        print("Download failed:", response.json())

def run_concurrent_backtests():
    """Run multiple backtests concurrently."""
    configs = [
        {"date": "2024-10-01", "base_target": 15.0},
        {"date": "2024-10-02", "base_target": 20.0},
        {"date": "2024-10-03", "base_target": 25.0},
    ]
    
    task_ids = []
    
    # Start all backtests
    for config in configs:
        payload = {**config, "run_async": True}
        response = requests.post(f"{BASE_URL}/backtest", json=payload)
        task_ids.append(response.json()["task_id"])
        print(f"Started backtest for {config['date']}: {response.json()['task_id']}")
    
    # Wait for all to complete
    completed = 0
    while completed < len(task_ids):
        for task_id in task_ids:
            status = requests.get(f"{BASE_URL}/backtest/{task_id}/status").json()
            if status['status'] in ['completed', 'failed']:
                completed += 1
                print(f"Task {task_id} completed with status: {status['status']}")
        time.sleep(1)
    
    # Get all results
    results = {}
    for task_id in task_ids:
        result = requests.get(f"{BASE_URL}/backtest/{task_id}/result").json()
        results[task_id] = result
    
    print("All results:", json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    print("Testing 5EMA Strategy Backtest API")
    print("="*50)
    
    # Test health
    test_health()
    
    # Run examples
    print("\n1. Running single day async...")
    run_single_day_async()
    
    print("\n2. Running date range sync...")
    run_date_range_sync()
    
    print("\n3. Running scalping config...")
    run_scalping_config()
    
    print("\n4. Listing all tasks...")
    list_all_tasks()
    
    print("\n5. Running concurrent backtests...")
    run_concurrent_backtests()
