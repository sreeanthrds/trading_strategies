# 5EMA Strategy Backtest API

FastAPI REST API for running 5EMA strategy backtests with dynamic configuration.

## Installation

```bash
pip install -r requirements.txt
```

## Starting the Server

```bash
python api_server.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Interactive docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### Health Check
```http
GET /health
```

### Start Backtest
```http
POST /backtest
Content-Type: application/json

{
  "date": "2024-10-03",
  "base_target": 25.0,
  "signal_start_time": "09:20",
  "force_exit_time": "15:25",
  "allow_opposite_entry": true,
  "daily_target": 100.0,
  "historical_candles": 500,
  "run_async": true,
  "save_results": true,
  "output_dir": "results"
}
```

### Get Task Status
```http
GET /backtest/{task_id}/status
```

### Get Backtest Result
```http
GET /backtest/{task_id}/result
```

### Download Results CSV
```http
GET /backtest/{task_id}/download
```

### List All Tasks
```http
GET /backtest/tasks
```

### Delete Task
```http
DELETE /backtest/{task_id}
```

## Usage Examples

### 1. Run Single Day (Synchronous)
```bash
curl -X POST "http://localhost:8000/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-10-03",
    "base_target": 20.0,
    "run_async": false
  }'
```

### 2. Run Date Range (Asynchronous)
```bash
curl -X POST "http://localhost:8000/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2024-10-01",
    "end_date": "2024-10-05",
    "base_target": 25.0,
    "run_async": true
  }'
```

### 3. Check Status
```bash
curl "http://localhost:8000/backtest/{task_id}/status"
```

### 4. Get Result
```bash
curl "http://localhost:8000/backtest/{task_id}/result"
```

### 5. Download CSV
```bash
curl -O "http://localhost:8000/backtest/{task_id}/download"
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date` | string | null | Single date to run (YYYY-MM-DD) |
| `start_date` | string | null | Start date for range (YYYY-MM-DD) |
| `end_date` | string | null | End date for range (YYYY-MM-DD) |
| `base_target` | float | 20.0 | Base target points |
| `signal_start_time` | string | "09:17" | Signal start time (HH:MM) |
| `force_exit_time` | string | "15:25" | Force exit time (HH:MM) |
| `allow_opposite_entry` | boolean | true | Allow opposite entry |
| `daily_target` | float | null | Daily target points |
| `historical_candles` | integer | 500 | Historical candles for warmup |
| `run_async` | boolean | false | Run in background |
| `save_results` | boolean | true | Save results to file |
| `output_dir` | string | "results" | Output directory |

## Python Client Examples

See `api_client_examples.py` for detailed Python client usage.

```python
import requests

# Start backtest
response = requests.post("http://localhost:8000/backtest", json={
    "date": "2024-10-03",
    "base_target": 25.0,
    "run_async": True
})

task_id = response.json()["task_id"]

# Check status
status = requests.get(f"http://localhost:8000/backtest/{task_id}/status").json()

# Get result
result = requests.get(f"http://localhost:8000/backtest/{task_id}/result").json()
```

## Predefined Configurations

### Conservative
```json
{
  "base_target": 15.0,
  "signal_start_time": "09:20",
  "force_exit_time": "15:20",
  "allow_opposite_entry": false,
  "daily_target": 50.0,
  "historical_candles": 500
}
```

### Aggressive
```json
{
  "base_target": 30.0,
  "signal_start_time": "09:15",
  "force_exit_time": "15:30",
  "allow_opposite_entry": true,
  "daily_target": null,
  "historical_candles": 300
}
```

### Scalping
```json
{
  "base_target": 10.0,
  "signal_start_time": "09:17",
  "force_exit_time": "15:25",
  "allow_opposite_entry": true,
  "daily_target": 100.0,
  "historical_candles": 200
}
```

### Swing
```json
{
  "base_target": 50.0,
  "signal_start_time": "09:30",
  "force_exit_time": "15:15",
  "allow_opposite_entry": false,
  "daily_target": null,
  "historical_candles": 1000
}
```

## Running with Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t 5ema-backtest-api .
docker run -p 8000:8000 5ema-backtest-api
```

## Monitoring

- Health check: `GET /health`
- Task status: `GET /backtest/{task_id}/status`
- List all tasks: `GET /backtest/tasks`

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Success
- 400: Bad request (invalid parameters)
- 404: Task not found
- 500: Internal server error (e.g., database connection failed)
