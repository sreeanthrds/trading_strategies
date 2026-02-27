"""
FastAPI Server with Lambda Integration
=====================================
FastAPI server that can trigger Lambda functions for distributed backtesting.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date, datetime, timedelta
import uuid
import json
import boto3
import os
from concurrent.futures import ThreadPoolExecutor

from config_loader import ExternalConfig, create_strategy_config
import orchestrator

app = FastAPI(
    title="5EMA Strategy Backtest API (Lambda)",
    description="FastAPI server with Lambda integration for distributed backtesting",
    version="1.0.0"
)

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
LAMBDA_FUNCTION_NAME = os.getenv('LAMBDA_FUNCTION_NAME', 'five-ema-backtest')
S3_BUCKET = os.getenv('S3_BUCKET', 'tradelayout-backtest-dev')
S3_PREFIX = os.getenv('S3_PREFIX', '5ema-results')

# Initialize AWS clients
lambda_client = boto3.client('lambda', region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)

# Global executor
executor = ThreadPoolExecutor(max_workers=4)

# Store for running tasks
running_tasks = {}

class LambdaBacktestRequest(BaseModel):
    """Backtest request for Lambda execution."""
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    
    # Strategy parameters
    base_target: Optional[float] = Field(20.0, description="Base target points")
    signal_start_time: Optional[str] = Field("09:17", description="Signal start time (HH:MM)")
    force_exit_time: Optional[str] = Field("15:25", description="Force exit time (HH:MM)")
    allow_opposite_entry: Optional[bool] = Field(True, description="Allow opposite entry")
    daily_target: Optional[float] = Field(None, description="Daily target points")
    historical_candles: Optional[int] = Field(500, description="Historical candles for warmup")
    
    # Lambda options
    async_execution: Optional[bool] = Field(True, description="Execute Lambda functions asynchronously")
    max_concurrent: Optional[int] = Field(10, description="Maximum concurrent Lambda invocations")
    save_results: Optional[bool] = Field(True, description="Save results to S3")
    send_telegram: Optional[bool] = Field(False, description="Send Telegram notification")

class LocalBacktestRequest(BaseModel):
    """Backtest request for local execution."""
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

class TaskStatus(BaseModel):
    """Task status model."""
    task_id: str
    execution_type: str  # 'lambda' or 'local'
    status: str  # pending, running, completed, failed
    progress: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

def run_lambda_backtest_task(task_id: str, request: LambdaBacktestRequest):
    """Run Lambda backtest in background."""
    try:
        # Update status
        running_tasks[task_id].status = "running"
        running_tasks[task_id].progress = "Initializing Lambda execution"
        
        # Create strategy config
        external_config = ExternalConfig(
            base_target=request.base_target,
            signal_start_time=request.signal_start_time,
            force_exit_time=request.force_exit_time,
            allow_opposite_entry=request.allow_opposite_entry,
            daily_target=request.daily_target,
            historical_candles=request.historical_candles,
        )
        config = create_strategy_config(external_config)
        
        # Convert to dict for Lambda payload
        lambda_config = {
            'base_target': config.base_target,
            'signal_start_time': config.signal_start_time,
            'force_exit_time': config.force_exit_time,
            'allow_opposite_entry': config.allow_opposite_entry,
            'daily_target': config.daily_target,
            'historical_candles': config.historical_candles,
        }
        
        # Update progress
        running_tasks[task_id].progress = "Running Lambda backtests"
        
        # Generate date range
        start_date = datetime.strptime(request.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(request.end_date, '%Y-%m-%d').date()
        
        dates = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Skip weekends
                dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        # Run backtest using orchestrator functions
        if request.async_execution:
            # Async execution
            results = []
            for date_str in dates:
                result = orchestrator.invoke_lambda_async(date_str, lambda_config)
                results.append(result)
            
            # For async, we can't get immediate results, so return invocation status
            successful_invocations = sum(1 for r in results if r['status'] == 'invoked')
            
            running_tasks[task_id].status = "completed"
            running_tasks[task_id].progress = "Lambda functions invoked"
            running_tasks[task_id].result_summary = {
                "total_days": len(dates),
                "successful_invocations": successful_invocations,
                "execution_type": "async",
                "note": "Results will be available in S3 after Lambda functions complete"
            }
        else:
            # Sync execution
            results = orchestrator.run_parallel_sync(dates, lambda_config, request.max_concurrent)
            
            # Calculate summary
            successful_days = sum(1 for r in results if r['status'] == 'success')
            total_trades = sum(r.get('trades', 0) for r in results)
            total_pnl = sum(r.get('pnl', 0) for r in results)
            target_hits = sum(r.get('target_hits', 0) for r in results)
            
            running_tasks[task_id].status = "completed"
            running_tasks[task_id].progress = "Completed"
            running_tasks[task_id].result_summary = {
                "total_days": len(dates),
                "successful_days": successful_days,
                "total_trades": total_trades,
                "total_pnl": total_pnl,
                "target_hits": target_hits,
                "execution_type": "sync",
                "results": results[:10]  # First 10 results for preview
            }
        
        running_tasks[task_id].completed_at = datetime.now()
        
    except Exception as e:
        # Update status with error
        running_tasks[task_id].status = "failed"
        running_tasks[task_id].progress = "Failed"
        running_tasks[task_id].error = str(e)
        running_tasks[task_id].completed_at = datetime.now()

def run_local_backtest_task(task_id: str, request: LocalBacktestRequest):
    """Run local backtest in background."""
    try:
        # Import here to avoid circular imports
        from five_ema_strategy import BacktestEngine, ClickHouseDataFetcher, ReportGenerator
        from datetime import timedelta
        import pandas as pd
        
        # Update status
        running_tasks[task_id].status = "running"
        running_tasks[task_id].progress = "Creating configuration"
        
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
        
        # Test connection
        running_tasks[task_id].progress = "Testing database connection"
        fetcher = ClickHouseDataFetcher(config)
        if not fetcher.test_connection():
            raise Exception("Cannot connect to ClickHouse")
        
        # Determine date range
        if request.date:
            run_date = datetime.strptime(request.date, '%Y-%m-%d').date()
            dates = [run_date]
        elif request.start_date and request.end_date:
            start_date = datetime.strptime(request.start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(request.end_date, '%Y-%m-%d').date()
            dates = []
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:  # Skip weekends
                    dates.append(current)
                current += timedelta(days=1)
        else:
            raise ValueError("Must provide either 'date' or 'start_date' and 'end_date'")
        
        # Run backtest
        engine = BacktestEngine(config)
        all_trades = []
        successful_days = 0
        
        for i, run_date in enumerate(dates):
            running_tasks[task_id].progress = f"Processing {run_date} ({i+1}/{len(dates)})"
            
            try:
                daily_trades = engine.run(run_date)
                if not daily_trades.empty:
                    all_trades.append(daily_trades)
                    successful_days += 1
            except Exception as e:
                print(f"Error on {run_date}: {e}")
        
        # Combine results
        if all_trades:
            trades_df = pd.concat(all_trades, ignore_index=True)
            
            # Calculate summary
            winners = trades_df[trades_df['trade_pnl'] > 0]
            losers = trades_df[trades_df['trade_pnl'] < 0]
            target_hits = trades_df[trades_df['exit_type'] == 'TARGET']
            
            summary = {
                "total_days": len(dates),
                "successful_days": successful_days,
                "total_trades": len(trades_df),
                "total_pnl": trades_df['cumulative_pnl'].iloc[-1] if not trades_df.empty else 0,
                "target_hits": len(target_hits),
                "winners": len(winners),
                "losers": len(losers),
                "total_profit": winners['trade_pnl'].sum() if not winners.empty else 0,
                "total_loss": losers['trade_pnl'].sum() if not losers.empty else 0,
            }
            
            # Save results
            if request.save_results:
                os.makedirs(request.output_dir, exist_ok=True)
                if request.date:
                    csv_path = f"{request.output_dir}/trades_{request.date}.csv"
                else:
                    csv_path = f"{request.output_dir}/trades_{request.start_date}_to_{request.end_date}.csv"
                trades_df.to_csv(csv_path, index=False)
                summary["result_file"] = csv_path
        else:
            summary = {
                "total_days": len(dates),
                "successful_days": 0,
                "total_trades": 0,
                "total_pnl": 0,
                "target_hits": 0,
                "winners": 0,
                "losers": 0,
                "total_profit": 0,
                "total_loss": 0,
            }
        
        # Update status
        running_tasks[task_id].status = "completed"
        running_tasks[task_id].progress = "Completed"
        running_tasks[task_id].result_summary = summary
        running_tasks[task_id].completed_at = datetime.now()
        
    except Exception as e:
        # Update status with error
        running_tasks[task_id].status = "failed"
        running_tasks[task_id].progress = "Failed"
        running_tasks[task_id].error = str(e)
        running_tasks[task_id].completed_at = datetime.now()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "5EMA Strategy Backtest API (Lambda + Local)",
        "version": "1.0.0",
        "execution_modes": ["lambda", "local"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test AWS credentials
        lambda_client.list_functions()
        
        # Test ClickHouse connection
        config = create_strategy_config()
        from five_ema_strategy import ClickHouseDataFetcher
        fetcher = ClickHouseDataFetcher(config)
        if fetcher.test_connection():
            return {
                "status": "healthy",
                "aws": "connected",
                "database": "connected"
            }
        else:
            return {
                "status": "unhealthy",
                "aws": "connected",
                "database": "disconnected"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "aws": "disconnected",
            "database": "unknown",
            "error": str(e)
        }

@app.post("/backtest/lambda", response_model=dict)
async def start_lambda_backtest(request: LambdaBacktestRequest, background_tasks: BackgroundTasks):
    """Start Lambda-based backtest."""
    # Generate task ID
    task_id = str(uuid.uuid4())[:8]
    
    # Create task status
    running_tasks[task_id] = TaskStatus(
        task_id=task_id,
        execution_type="lambda",
        status="pending",
        created_at=datetime.now()
    )
    
    # Run in background
    background_tasks.add_task(run_lambda_backtest_task, task_id, request)
    
    return {"task_id": task_id, "status": "started", "message": "Lambda backtest started in background"}

@app.post("/backtest/local", response_model=dict)
async def start_local_backtest(request: LocalBacktestRequest, background_tasks: BackgroundTasks):
    """Start local backtest."""
    # Validate input
    if not request.date and not (request.start_date and request.end_date):
        raise HTTPException(status_code=400, detail="Must provide either 'date' or 'start_date' and 'end_date'")
    
    # Generate task ID
    task_id = str(uuid.uuid4())[:8]
    
    # Create task status
    running_tasks[task_id] = TaskStatus(
        task_id=task_id,
        execution_type="local",
        status="pending",
        created_at=datetime.now()
    )
    
    # Run in background
    background_tasks.add_task(run_local_backtest_task, task_id, request)
    
    return {"task_id": task_id, "status": "started", "message": "Local backtest started in background"}

@app.get("/backtest/{task_id}/status", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get task status."""
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return running_tasks[task_id]

@app.get("/backtest/tasks", response_model=List[TaskStatus])
async def list_tasks():
    """List all tasks."""
    return list(running_tasks.values())

@app.delete("/backtest/{task_id}")
async def delete_task(task_id: str):
    """Delete a task."""
    if task_id in running_tasks:
        del running_tasks[task_id]
        return {"message": "Task deleted"}
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@app.get("/lambda/functions")
async def list_lambda_functions():
    """List available Lambda functions."""
    try:
        functions = lambda_client.list_functions()
        return {"functions": [f['FunctionName'] for f in functions['Functions']]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/s3/results")
async def list_s3_results():
    """List S3 results."""
    try:
        objects = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
        results = []
        for obj in objects.get('Contents', []):
            results.append({
                "key": obj['Key'],
                "size": obj['Size'],
                "last_modified": obj['LastModified']
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
