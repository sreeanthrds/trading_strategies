"""
FastAPI Server for 5EMA Strategy Backtesting
=============================================
REST API to trigger backtests with dynamic configuration.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime
import os
import uuid
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config_loader import ExternalConfig, create_strategy_config, print_config_summary
from five_ema_strategy import BacktestEngine, ClickHouseDataFetcher, ReportGenerator

app = FastAPI(
    title="5EMA Strategy Backtest API",
    description="REST API for running 5EMA strategy backtests with dynamic configuration",
    version="1.0.0"
)

# Global executor for background tasks
executor = ThreadPoolExecutor(max_workers=4)

# Store for running tasks
running_tasks = {}

class BacktestRequest(BaseModel):
    """Backtest request model."""
    date: Optional[str] = Field(None, description="Single date to run (YYYY-MM-DD)")
    start_date: Optional[str] = Field(None, description="Start date for range (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date for range (YYYY-MM-DD)")
    
    # Strategy parameters
    base_target: Optional[float] = Field(20.0, description="Base target points")
    signal_start_time: Optional[str] = Field("09:17", description="Signal start time (HH:MM)")
    force_exit_time: Optional[str] = Field("15:25", description="Force exit time (HH:MM)")
    allow_opposite_entry: Optional[bool] = Field(True, description="Allow opposite entry")
    daily_target: Optional[float] = Field(None, description="Daily target points")
    historical_candles: Optional[int] = Field(500, description="Historical candles for warmup")
    
    # Output options
    save_results: Optional[bool] = Field(True, description="Save results to file")
    output_dir: Optional[str] = Field("results", description="Output directory")
    run_async: Optional[bool] = Field(False, description="Run in background")

class BacktestStatus(BaseModel):
    """Backtest status model."""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: Optional[str] = None
    result_file: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime

class BacktestResult(BaseModel):
    """Backtest result model."""
    task_id: str
    status: str
    total_trades: Optional[int] = None
    total_pnl: Optional[float] = None
    target_hits: Optional[int] = None
    winners: Optional[int] = None
    losers: Optional[int] = None
    days_processed: Optional[int] = None
    result_file: Optional[str] = None
    summary: Optional[dict] = None

def run_backtest_task(task_id: str, request: BacktestRequest):
    """Run backtest in background."""
    try:
        # Update status
        running_tasks[task_id].status = "running"
        
        # Create configuration
        external_config = ExternalConfig(
            base_target=request.base_target,
            signal_start_time=request.signal_start_time,
            force_exit_time=request.force_exit_time,
            allow_opposite_entry=request.allow_opposite_entry,
            daily_target=request.daily_target,
            historical_candles=request.historical_candles,
        )
        config = create_strategy_config(external_config)
        
        # Determine date range
        if request.date:
            run_date = datetime.strptime(request.date, '%Y-%m-%d').date()
            trades_df = run_single_date(config, run_date, request.output_dir, request.save_results)
            days_processed = 1
        elif request.start_date and request.end_date:
            start_date = datetime.strptime(request.start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(request.end_date, '%Y-%m-%d').date()
            trades_df = run_date_range(config, start_date, end_date, request.output_dir, request.save_results)
            days_processed = (end_date - start_date).days + 1
        else:
            raise ValueError("Must provide either 'date' or 'start_date' and 'end_date'")
        
        # Calculate summary
        summary = {}
        if trades_df is not None and not trades_df.empty:
            summary = {
                'total_trades': len(trades_df),
                'total_pnl': trades_df['cumulative_pnl'].iloc[-1],
                'target_hits': len(trades_df[trades_df['exit_type'] == 'TARGET']),
                'winners': len(trades_df[trades_df['trade_pnl'] > 0]),
                'losers': len(trades_df[trades_df['trade_pnl'] < 0]),
            }
        
        # Update status
        task = running_tasks[task_id]
        task.status = "completed"
        task.result_file = f"{request.output_dir}/trades_{request.date or request.start_date + '_to_' + request.end_date}.csv" if request.save_results else None
        
        # Store result
        running_tasks[f"{task_id}_result"] = BacktestResult(
            task_id=task_id,
            status="completed",
            days_processed=days_processed,
            result_file=task.result_file,
            **summary
        )
        
    except Exception as e:
        # Update status with error
        running_tasks[task_id].status = "failed"
        running_tasks[task_id].error = str(e)
        running_tasks[f"{task_id}_result"] = BacktestResult(
            task_id=task_id,
            status="failed",
            error=str(e)
        )

def run_single_date(config, run_date, output_dir, save_results):
    """Run backtest for a single date."""
    print(f"Running backtest for {run_date}")
    
    # Test connection
    fetcher = ClickHouseDataFetcher(config)
    if not fetcher.test_connection():
        raise Exception("Cannot connect to ClickHouse")
    
    # Run backtest
    engine = BacktestEngine(config)
    trades_df = engine.run(run_date)
    
    if trades_df.empty:
        print(f"No trades on {run_date}")
        return None
    
    # Save results
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = f"{output_dir}/trades_{run_date}.csv"
        trades_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    
    return trades_df

def run_date_range(config, start_date, end_date, output_dir, save_results):
    """Run backtest for a date range."""
    from datetime import timedelta
    import pandas as pd
    
    print(f"Running backtest from {start_date} to {end_date}")
    
    # Test connection
    fetcher = ClickHouseDataFetcher(config)
    if not fetcher.test_connection():
        raise Exception("Cannot connect to ClickHouse")
    
    # Collect all trades
    engine = BacktestEngine(config)
    all_trades = []
    current_date = start_date
    total_days = 0
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() < 5:
            total_days += 1
            print(f"Processing {current_date}...")
            
            try:
                daily_trades = engine.run(current_date)
                if not daily_trades.empty:
                    all_trades.append(daily_trades)
            except Exception as e:
                print(f"Error on {current_date}: {e}")
        
        current_date += timedelta(days=1)
    
    # Combine results
    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            csv_path = f"{output_dir}/trades_{start_date}_to_{end_date}.csv"
            trades_df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")
        
        return trades_df
    else:
        return None

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "5EMA Strategy Backtest API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test ClickHouse connection
        config = create_strategy_config()
        fetcher = ClickHouseDataFetcher(config)
        if fetcher.test_connection():
            return {"status": "healthy", "database": "connected"}
        else:
            return {"status": "unhealthy", "database": "disconnected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/backtest", response_model=dict)
async def start_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Start a new backtest."""
    # Validate input
    if not request.date and not (request.start_date and request.end_date):
        raise HTTPException(status_code=400, detail="Must provide either 'date' or 'start_date' and 'end_date'")
    
    # Generate task ID
    task_id = str(uuid.uuid4())[:8]
    
    # Create task status
    running_tasks[task_id] = BacktestStatus(
        task_id=task_id,
        status="pending",
        created_at=datetime.now()
    )
    
    # Run backtest
    if request.run_async:
        # Run in background
        background_tasks.add_task(run_backtest_task, task_id, request)
        return {"task_id": task_id, "status": "started", "message": "Backtest started in background"}
    else:
        # Run synchronously
        try:
            run_backtest_task(task_id, request)
            result = running_tasks.get(f"{task_id}_result")
            if result.status == "failed":
                raise HTTPException(status_code=500, detail=result.error)
            return {"task_id": task_id, "status": "completed", "result": result.dict()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest/{task_id}/status", response_model=BacktestStatus)
async def get_backtest_status(task_id: str):
    """Get backtest status."""
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return running_tasks[task_id]

@app.get("/backtest/{task_id}/result", response_model=BacktestResult)
async def get_backtest_result(task_id: str):
    """Get backtest result."""
    result_key = f"{task_id}_result"
    if result_key not in running_tasks:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return running_tasks[result_key]

@app.get("/backtest/{task_id}/download")
async def download_backtest_result(task_id: str):
    """Download backtest result CSV."""
    result_key = f"{task_id}_result"
    if result_key not in running_tasks:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result = running_tasks[result_key]
    if not result.result_file or not os.path.exists(result.result_file):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        result.result_file,
        media_type="text/csv",
        filename=os.path.basename(result.result_file)
    )

@app.get("/backtest/tasks", response_model=List[BacktestStatus])
async def list_tasks():
    """List all tasks."""
    return [task for task_id, task in running_tasks.items() if not task_id.endswith("_result")]

@app.delete("/backtest/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its results."""
    if task_id in running_tasks:
        del running_tasks[task_id]
    
    result_key = f"{task_id}_result"
    if result_key in running_tasks:
        result = running_tasks[result_key]
        if result.result_file and os.path.exists(result.result_file):
            os.remove(result.result_file)
        del running_tasks[result_key]
    
    return {"message": "Task deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
