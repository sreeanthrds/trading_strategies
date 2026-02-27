"""
FastAPI + Lambda Integration Examples
=====================================
Examples of using the hybrid API with Lambda and local execution.
"""

import requests
import json
import time
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test API health with AWS and database status."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    return response.json()

def run_lambda_backtest():
    """Run backtest using Lambda functions."""
    payload = {
        "start_date": "2024-10-01",
        "end_date": "2024-10-05",
        "base_target": 25.0,
        "signal_start_time": "09:20",
        "daily_target": 100.0,
        "async_execution": True,
        "max_concurrent": 5,
        "save_results": True,
        "send_telegram": False
    }
    
    response = requests.post(f"{BASE_URL}/backtest/lambda", json=payload)
    print("Started Lambda backtest:", response.json())
    
    task_id = response.json()["task_id"]
    
    # Poll for completion
    while True:
        status = requests.get(f"{BASE_URL}/backtest/{task_id}/status").json()
        print(f"Status: {status['status']} | Progress: {status.get('progress', 'N/A')}")
        
        if status['status'] in ['completed', 'failed']:
            break
        
        time.sleep(5)
    
    # Get final result
    result = requests.get(f"{BASE_URL}/backtest/{task_id}/status").json()
    print("\nLambda Backtest Result:")
    print(json.dumps(result, indent=2))
    
    return result

def run_local_backtest():
    """Run backtest locally."""
    payload = {
        "start_date": "2024-10-01",
        "end_date": "2024-10-03",
        "base_target": 20.0,
        "allow_opposite_entry": True,
        "historical_candles": 300,
        "save_results": True,
        "output_dir": "local_results"
    }
    
    response = requests.post(f"{BASE_URL}/backtest/local", json=payload)
    print("Started Local backtest:", response.json())
    
    task_id = response.json()["task_id"]
    
    # Poll for completion
    while True:
        status = requests.get(f"{BASE_URL}/backtest/{task_id}/status").json()
        print(f"Status: {status['status']} | Progress: {status.get('progress', 'N/A')}")
        
        if status['status'] in ['completed', 'failed']:
            break
        
        time.sleep(2)
    
    # Get final result
    result = requests.get(f"{BASE_URL}/backtest/{task_id}/status").json()
    print("\nLocal Backtest Result:")
    print(json.dumps(result, indent=2))
    
    return result

def compare_execution_modes():
    """Compare Lambda vs Local execution for same period."""
    test_config = {
        "start_date": "2024-10-01",
        "end_date": "2024-10-02",
        "base_target": 20.0,
        "signal_start_time": "09:17",
        "force_exit_time": "15:25",
        "allow_opposite_entry": True,
        "daily_target": None,
        "historical_candles": 500
    }
    
    print("Running Lambda execution...")
    lambda_response = requests.post(f"{BASE_URL}/backtest/lambda", json={
        **test_config,
        "async_execution": True,
        "max_concurrent": 2,
        "save_results": True,
        "send_telegram": False
    })
    lambda_task_id = lambda_response.json()["task_id"]
    
    print("Running Local execution...")
    local_response = requests.post(f"{BASE_URL}/backtest/local", json={
        **test_config,
        "save_results": True,
        "output_dir": "comparison_local"
    })
    local_task_id = local_response.json()["task_id"]
    
    # Wait for both to complete
    results = {}
    
    for name, task_id in [("Lambda", lambda_task_id), ("Local", local_task_id)]:
        print(f"\nWaiting for {name} execution...")
        while True:
            status = requests.get(f"{BASE_URL}/backtest/{task_id}/status").json()
            print(f"{name}: {status['status']} - {status.get('progress', 'N/A')}")
            
            if status['status'] in ['completed', 'failed']:
                results[name] = status
                break
            
            time.sleep(2)
    
    # Compare results
    print("\n" + "="*60)
    print("EXECUTION COMPARISON")
    print("="*60)
    
    for name, result in results.items():
        if result['status'] == 'completed':
            summary = result.get('result_summary', {})
            print(f"\n{name} Results:")
            print(f"  Total Trades: {summary.get('total_trades', 0)}")
            print(f"  Total P&L: {summary.get('total_pnl', 0):+.2f} pts")
            print(f"  Target Hits: {summary.get('target_hits', 0)}")
            print(f"  Execution Time: {summary.get('execution_time', 0):.2f}s")
        else:
            print(f"\n{name} Failed: {result.get('error', 'Unknown error')}")
    
    return results

def run_concurrent_tests():
    """Run multiple concurrent tests."""
    configs = [
        {
            "name": "Conservative",
            "base_target": 15.0,
            "signal_start_time": "09:20",
            "force_exit_time": "15:20",
            "allow_opposite_entry": False
        },
        {
            "name": "Aggressive",
            "base_target": 30.0,
            "signal_start_time": "09:15",
            "force_exit_time": "15:30",
            "allow_opposite_entry": True
        },
        {
            "name": "Scalping",
            "base_target": 10.0,
            "signal_start_time": "09:17",
            "force_exit_time": "15:25",
            "allow_opposite_entry": True,
            "daily_target": 100.0
        }
    ]
    
    task_ids = []
    
    # Start all tests
    for config in configs:
        payload = {
            "start_date": "2024-10-01",
            "end_date": "2024-10-02",
            **config,
            "async_execution": True,
            "max_concurrent": 2,
            "save_results": True,
            "send_telegram": False
        }
        
        response = requests.post(f"{BASE_URL}/backtest/lambda", json=payload)
        task_id = response.json()["task_id"]
        task_ids.append((config["name"], task_id))
        print(f"Started {config['name']} test: {task_id}")
    
    # Wait for all to complete
    completed = 0
    while completed < len(task_ids):
        for name, task_id in task_ids:
            status = requests.get(f"{BASE_URL}/backtest/{task_id}/status").json()
            if status['status'] in ['completed', 'failed']:
                completed += 1
                print(f"{name} completed with status: {status['status']}")
        time.sleep(2)
    
    # Get all results
    results = {}
    for name, task_id in task_ids:
        result = requests.get(f"{BASE_URL}/backtest/{task_id}/status").json()
        results[name] = result
    
    print("\n" + "="*60)
    print("CONCURRENT TEST RESULTS")
    print("="*60)
    
    for name, result in results.items():
        if result['status'] == 'completed':
            summary = result.get('result_summary', {})
            print(f"\n{name}:")
            print(f"  P&L: {summary.get('total_pnl', 0):+.2f} pts")
            print(f"  Trades: {summary.get('total_trades', 0)}")
            print(f"  Targets: {summary.get('target_hits', 0)}")
            print(f"  Win Rate: {(summary.get('winners', 0)/max(summary.get('total_trades', 1), 1)*100):.1f}%")
    
    return results

def list_aws_resources():
    """List available AWS resources."""
    print("\nAvailable Lambda Functions:")
    lambda_funcs = requests.get(f"{BASE_URL}/lambda/functions").json()
    for func in lambda_funcs.get('functions', []):
        print(f"  - {func}")
    
    print("\nS3 Results:")
    s3_results = requests.get(f"{BASE_URL}/s3/results").json()
    for result in s3_results.get('results', [])[:10]:  # Show first 10
        print(f"  - {result['key']} ({result['size']} bytes)")
    
    return lambda_funcs, s3_results

if __name__ == "__main__":
    print("Testing FastAPI + Lambda Integration")
    print("="*50)
    
    # Test health
    test_health()
    
    # List AWS resources
    list_aws_resources()
    
    # Run Lambda backtest
    print("\n1. Running Lambda backtest...")
    run_lambda_backtest()
    
    # Run Local backtest
    print("\n2. Running Local backtest...")
    run_local_backtest()
    
    # Compare execution modes
    print("\n3. Comparing execution modes...")
    compare_execution_modes()
    
    # Run concurrent tests
    print("\n4. Running concurrent tests...")
    run_concurrent_tests()
