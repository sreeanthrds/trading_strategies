"""
AWS Lambda Handler for 5 EMA Strategy Single-Day Backtest
=========================================================
Each Lambda invocation processes ONE trading day:
1. Loads OHLCV + tick data from ClickHouse
2. Runs full backtest with tick-level precision
3. Saves results CSV to S3
4. Sends Telegram notification with summary

Expected event payload:
{
    "date": "2024-10-03",
    "symbol": "NIFTY",
    "config": {  // optional overrides
        "base_target": 20.0,
        "signal_start_time": "09:21",
        ...
    },
    "s3_bucket": "5ema-backtest-results",
    "telegram_chat_id": "6343050453"
}
"""

import os
import json
import time
import boto3
import requests as http_requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import strategy classes from the main module
# In Lambda, both files are in the same directory
from five_ema_strategy import (
    StrategyConfig, ClickHouseDataFetcher, SignalDetector,
    TickPrecisionEngine, PositionManager, BacktestEngine, ReportGenerator
)


# ============================================================================
# TELEGRAM NOTIFICATION
# ============================================================================

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')

def send_telegram(chat_id: str, message: str):
    """Send a message via Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN:
        print(f"[Telegram] No bot token configured, skipping: {message[:100]}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML',
        }
        resp = http_requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print(f"[Telegram] Sent to {chat_id}")
        else:
            print(f"[Telegram] Error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[Telegram] Failed: {e}")


# ============================================================================
# S3 HELPER
# ============================================================================

s3_client = boto3.client('s3')

def save_to_s3(bucket: str, key: str, data: str):
    """Save string data to S3."""
    s3_client.put_object(Bucket=bucket, Key=key, Body=data.encode('utf-8'))
    print(f"[S3] Saved to s3://{bucket}/{key}")


def save_df_to_s3(bucket: str, key: str, df: pd.DataFrame):
    """Save DataFrame as CSV to S3."""
    csv_data = df.to_csv(index=False)
    save_to_s3(bucket, key, csv_data)


# ============================================================================
# LAMBDA HANDLER
# ============================================================================

def lambda_handler(event, context):
    """
    Main Lambda entry point. Processes a single trading day.

    Returns:
    {
        "date": "2024-10-03",
        "status": "success" | "error" | "no_data",
        "trades": 60,
        "pnl": 15.95,
        "winners": 8,
        "losers": 52,
        "duration_seconds": 5.2,
        "s3_key": "results/2024-10-03.csv",
        "error": null
    }
    """
    start_time = time.time()

    # Parse event
    date_str = event.get('date')
    symbol = event.get('symbol', 'NIFTY')
    config_overrides = event.get('config', {})
    s3_bucket = event.get('s3_bucket', os.environ.get('S3_BUCKET', '5ema-backtest-results'))
    s3_prefix = event.get('s3_prefix', 'results')
    telegram_chat_id = event.get('telegram_chat_id', os.environ.get('TELEGRAM_CHAT_ID', '6343050453'))

    result = {
        'date': date_str,
        'symbol': symbol,
        'status': 'error',
        'trades': 0,
        'pnl': 0.0,
        'winners': 0,
        'losers': 0,
        'force_exits': 0,
        'target_hits': 0,
        'total_profit': 0.0,
        'total_loss': 0.0,
        'duration_seconds': 0.0,
        's3_key': None,
        'error': None,
    }

    try:
        if not date_str:
            raise ValueError("Missing 'date' in event payload")

        trade_date = datetime.strptime(date_str, '%Y-%m-%d').date()

        # Build config
        config = StrategyConfig(
            base_target=config_overrides.get('base_target', 20.0),
            signal_start_time=config_overrides.get('signal_start_time', '09:17'),
            force_exit_time=config_overrides.get('force_exit_time', '15:25'),
            allow_opposite_entry=config_overrides.get('allow_opposite_entry', True),
            daily_target=config_overrides.get('daily_target', None),
            historical_candles=config_overrides.get('historical_candles', 500),
            # ClickHouse connection from env or defaults
            clickhouse_host=os.environ.get('CLICKHOUSE_HOST', '34.200.220.45'),
            clickhouse_port=os.environ.get('CLICKHOUSE_PORT', '8123'),
            clickhouse_user=os.environ.get('CLICKHOUSE_USER', 'default'),
            clickhouse_password=os.environ.get('CLICKHOUSE_PASSWORD', ''),
            clickhouse_database=os.environ.get('CLICKHOUSE_DATABASE', 'tradelayout'),
        )

        # Run backtest
        engine = BacktestEngine(config)
        trades_df = engine.run(trade_date, symbol)

        elapsed = time.time() - start_time

        if trades_df.empty:
            result['status'] = 'no_data'
            result['duration_seconds'] = round(elapsed, 1)

            # Telegram notification
            msg = f"📊 <b>{date_str}</b> | No trades | ⏱ {elapsed:.1f}s"
            send_telegram(telegram_chat_id, msg)
        else:
            # Compute summary
            total = len(trades_df)
            winners = int((trades_df['trade_pnl'] > 0).sum())
            losers = int((trades_df['trade_pnl'] < 0).sum())
            force_exits = int((trades_df['exit_type'] == 'FORCE_EXIT').sum())
            target_hits = int((trades_df['exit_type'] == 'TARGET').sum())
            final_pnl = float(trades_df['cumulative_pnl'].iloc[-1])
            total_profit = float(trades_df[trades_df['trade_pnl'] > 0]['trade_pnl'].sum())
            total_loss = float(trades_df[trades_df['trade_pnl'] < 0]['trade_pnl'].sum())

            result['status'] = 'success'
            result['trades'] = total
            result['pnl'] = round(final_pnl, 2)
            result['winners'] = winners
            result['losers'] = losers
            result['force_exits'] = force_exits
            result['target_hits'] = target_hits
            result['total_profit'] = round(total_profit, 2)
            result['total_loss'] = round(total_loss, 2)
            result['duration_seconds'] = round(elapsed, 1)

            # Save to S3
            s3_key = f"{s3_prefix}/{date_str}.csv"
            save_df_to_s3(s3_bucket, s3_key, trades_df)
            result['s3_key'] = s3_key

            # Also save summary JSON
            summary_key = f"{s3_prefix}/{date_str}_summary.json"
            save_to_s3(s3_bucket, summary_key, json.dumps(result, indent=2))

            # Telegram notification
            pnl_emoji = "✅" if final_pnl > 0 else "❌"
            msg = (
                f"{pnl_emoji} <b>{date_str}</b> | "
                f"Trades: {total} | "
                f"Targets: {target_hits} | "
                f"P&L: <b>{final_pnl:+.2f}</b> pts | "
                f"W/L: {winners}/{losers} | "
                f"⏱ {elapsed:.1f}s"
            )
            send_telegram(telegram_chat_id, msg)

    except Exception as e:
        elapsed = time.time() - start_time
        result['status'] = 'error'
        result['error'] = str(e)
        result['duration_seconds'] = round(elapsed, 1)

        # Telegram error notification
        msg = f"🚨 <b>{date_str}</b> | Error: {str(e)[:200]} | ⏱ {elapsed:.1f}s"
        send_telegram(telegram_chat_id, msg)

        print(f"[ERROR] {date_str}: {e}")
        import traceback
        traceback.print_exc()

    print(f"[RESULT] {json.dumps(result)}")
    return result
