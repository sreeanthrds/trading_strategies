"""
AWS Lambda Orchestrator for 5 EMA Multi-Day Backtest
=====================================================
Invokes Lambda functions in parallel for each trading day,
monitors completion, aggregates results from S3, and sends
final summary via Telegram.

Usage:
    python orchestrator.py --start 2024-06-01 --end 2024-12-31
    python orchestrator.py --start 2024-10-01 --end 2024-10-05 --dry-run
"""

import os
import json
import time
import argparse
import boto3
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

# ============================================================================
# CONFIGURATION
# ============================================================================

AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
LAMBDA_FUNCTION_NAME = os.environ.get('LAMBDA_FUNCTION_NAME', '5ema-backtest')
S3_BUCKET = os.environ.get('S3_BUCKET', 'tradelayout-backtest-dev')
S3_PREFIX = '5ema-results'
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '6343050453')

# AWS clients
lambda_client = boto3.client('lambda', region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)


# ============================================================================
# TELEGRAM
# ============================================================================

def send_telegram(message: str):
    """Send notification via Telegram."""
    if not TELEGRAM_BOT_TOKEN:
        print(f"[Telegram] No token: {message[:100]}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML',
        }, timeout=10)
        if resp.status_code != 200:
            print(f"[Telegram] Error: {resp.text[:200]}")
    except Exception as e:
        print(f"[Telegram] Failed: {e}")


# ============================================================================
# DATE GENERATION
# ============================================================================

def generate_trading_dates(start_date: str, end_date: str) -> list:
    """Generate list of weekday dates (skip weekends)."""
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()

    dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday=0, Friday=4
            dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    return dates


# ============================================================================
# LAMBDA INVOCATION
# ============================================================================

def invoke_lambda(date_str: str, config: dict = None) -> dict:
    """Invoke Lambda function for a single date. Returns result dict."""
    payload = {
        'date': date_str,
        'symbol': 'NIFTY',
        's3_bucket': S3_BUCKET,
        's3_prefix': S3_PREFIX,
        'telegram_chat_id': TELEGRAM_CHAT_ID,
    }
    if config:
        payload['config'] = config

    try:
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType='RequestResponse',  # Synchronous
            Payload=json.dumps(payload),
        )

        # Parse response
        response_payload = json.loads(response['Payload'].read().decode('utf-8'))

        if response.get('FunctionError'):
            return {
                'date': date_str,
                'status': 'lambda_error',
                'error': response_payload.get('errorMessage', 'Unknown error'),
                'trades': 0,
                'pnl': 0.0,
            }

        return response_payload

    except Exception as e:
        return {
            'date': date_str,
            'status': 'invocation_error',
            'error': str(e),
            'trades': 0,
            'pnl': 0.0,
        }


def invoke_lambda_async(date_str: str, config: dict = None) -> dict:
    """Invoke Lambda function asynchronously (fire-and-forget)."""
    payload = {
        'date': date_str,
        'symbol': 'NIFTY',
        's3_bucket': S3_BUCKET,
        's3_prefix': S3_PREFIX,
        'telegram_chat_id': TELEGRAM_CHAT_ID,
    }
    if config:
        payload['config'] = config

    try:
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType='Event',  # Asynchronous
            Payload=json.dumps(payload),
        )
        status_code = response.get('StatusCode', 0)
        if status_code == 202:
            return {'date': date_str, 'status': 'invoked'}
        else:
            return {'date': date_str, 'status': 'invoke_failed', 'error': f'StatusCode: {status_code}'}

    except Exception as e:
        return {'date': date_str, 'status': 'invocation_error', 'error': str(e)}


# ============================================================================
# PARALLEL EXECUTION (Synchronous with ThreadPool)
# ============================================================================

def run_parallel_sync(dates: list, config: dict = None, max_workers: int = 50) -> list:
    """
    Invoke Lambda functions in parallel using thread pool.
    Each thread waits for its Lambda to complete (synchronous invocation).
    Returns list of result dicts.
    """
    results = []
    total = len(dates)

    print(f"\nInvoking {total} Lambda functions (max {max_workers} concurrent)...")
    send_telegram(f"🚀 Starting backtest for <b>{total}</b> trading days")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(invoke_lambda, date_str, config): date_str
            for date_str in dates
        }

        completed = 0
        for future in as_completed(futures):
            date_str = futures[future]
            completed += 1

            try:
                result = future.result()
                results.append(result)

                status = result.get('status', 'unknown')
                pnl = result.get('pnl', 0)
                trades = result.get('trades', 0)
                duration = result.get('duration_seconds', 0)

                status_icon = '✅' if status == 'success' and pnl > 0 else '❌' if status == 'success' else '⚠️'
                print(f"  [{completed}/{total}] {status_icon} {date_str} | "
                      f"Trades: {trades} | P&L: {pnl:+.2f} | {duration:.1f}s")

            except Exception as e:
                results.append({
                    'date': date_str,
                    'status': 'thread_error',
                    'error': str(e),
                    'trades': 0,
                    'pnl': 0.0,
                })
                print(f"  [{completed}/{total}] 🚨 {date_str} | Error: {e}")

    return results


# ============================================================================
# FIRE-AND-FORGET EXECUTION (Asynchronous)
# ============================================================================

def run_async_all(dates: list, config: dict = None) -> list:
    """
    Fire all Lambda functions asynchronously (Event invocation).
    Each Lambda handles its own Telegram notification and S3 storage.
    Returns list of invocation statuses.
    """
    results = []
    total = len(dates)

    print(f"\nFiring {total} async Lambda invocations...")
    send_telegram(f"🚀 Firing <b>{total}</b> backtest Lambda functions (async)")

    for i, date_str in enumerate(dates):
        result = invoke_lambda_async(date_str, config)
        results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  Invoked {i + 1}/{total}...")
            time.sleep(0.1)  # Small delay to avoid throttling

    invoked = sum(1 for r in results if r['status'] == 'invoked')
    failed = sum(1 for r in results if r['status'] != 'invoked')
    print(f"\nInvoked: {invoked}, Failed: {failed}")

    return results


# ============================================================================
# S3 AGGREGATION
# ============================================================================

def aggregate_results_from_s3(dates: list) -> pd.DataFrame:
    """Download and combine all daily result CSVs from S3."""
    print(f"\nAggregating results from S3...")
    all_dfs = []

    for date_str in dates:
        key = f"{S3_PREFIX}/{date_str}.csv"
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            csv_data = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_data))
            all_dfs.append(df)
        except s3_client.exceptions.NoSuchKey:
            pass  # No trades for this date
        except Exception as e:
            print(f"  Error reading {key}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"  Combined {len(all_dfs)} daily files → {len(combined)} total trades")
        return combined

    return pd.DataFrame()


def download_summaries_from_s3(dates: list) -> list:
    """Download all daily summary JSONs from S3."""
    summaries = []
    for date_str in dates:
        key = f"{S3_PREFIX}/{date_str}_summary.json"
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            summaries.append(data)
        except:
            pass
    return summaries


# ============================================================================
# REPORTING
# ============================================================================

def print_final_summary(results: list):
    """Print aggregated summary of all Lambda results."""
    successful = [r for r in results if r.get('status') == 'success']
    no_data = [r for r in results if r.get('status') == 'no_data']
    errors = [r for r in results if r.get('status') not in ('success', 'no_data')]

    total_trades = sum(r.get('trades', 0) for r in successful)
    total_pnl = sum(r.get('pnl', 0) for r in successful)
    total_winners = sum(r.get('winners', 0) for r in successful)
    total_losers = sum(r.get('losers', 0) for r in successful)
    total_target_hits = sum(r.get('target_hits', 0) for r in successful)
    total_profit = sum(r.get('total_profit', 0) for r in successful)
    total_loss = sum(r.get('total_loss', 0) for r in successful)
    days_with_targets = sum(1 for r in successful if r.get('target_hits', 0) > 0)
    days_without_targets = sum(1 for r in successful if r.get('target_hits', 0) == 0)
    avg_duration = (sum(r.get('duration_seconds', 0) for r in results) / len(results)
                    if results else 0)

    print(f"\n{'='*70}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*70}")
    print(f"  Days processed       : {len(results)}")
    print(f"  Successful           : {len(successful)}")
    print(f"  No data              : {len(no_data)}")
    print(f"  Errors               : {len(errors)}")
    print(f"  Total trades         : {total_trades}")
    print(f"  Total Target Hits    : {total_target_hits}")
    print(f"  Days with Targets    : {days_with_targets}")
    print(f"  Days with 0 Targets  : {days_without_targets}")
    print(f"  Total P&L            : {total_pnl:+.2f} pts")
    print(f"  Total Profit         : {total_profit:+.2f} pts")
    print(f"  Total Loss           : {total_loss:+.2f} pts")
    print(f"  Winners/Losers       : {total_winners}/{total_losers}")
    print(f"  Avg duration         : {avg_duration:.1f}s per day")
    print(f"{'='*70}")

    # Daily breakdown table
    print(f"\n  {'Date':<14} {'Trades':>7} {'Targets':>8} {'P&L':>10}")
    print(f"  {'─'*42}")
    for r in sorted(successful, key=lambda x: x['date']):
        print(f"  {r['date']:<14} {r.get('trades',0):>7} "
              f"{r.get('target_hits',0):>8} {r.get('pnl',0):>+10.2f}")

    if errors:
        print(f"\n  Failed dates:")
        for r in errors:
            print(f"    {r['date']}: {r.get('error', 'unknown')}")

    # Telegram summary
    pnl_emoji = "📈" if total_pnl > 0 else "📉"
    summary_msg = (
        f"{pnl_emoji} <b>BACKTEST COMPLETE</b>\n\n"
        f"📅 Days: {len(successful)} successful, {len(errors)} errors\n"
        f"📊 Trades: {total_trades} | Targets: {total_target_hits}\n"
        f"🎯 Days with Targets: {days_with_targets} | Without: {days_without_targets}\n"
        f"💰 Total P&L: <b>{total_pnl:+.2f}</b> pts\n"
        f"📈 Profit: {total_profit:+.2f} | Loss: {total_loss:+.2f}\n"
        f"✅ Winners: {total_winners} | ❌ Losers: {total_losers}\n"
        f"⏱ Avg: {avg_duration:.1f}s/day"
    )
    send_telegram(summary_msg)

    return {
        'total_days': len(results),
        'successful': len(successful),
        'errors': len(errors),
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'total_target_hits': total_target_hits,
        'days_with_targets': days_with_targets,
        'days_without_targets': days_without_targets,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='5 EMA Backtest Orchestrator')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--mode', choices=['sync', 'async'], default='async',
                        help='sync: wait for results; async: fire-and-forget')
    parser.add_argument('--max-workers', type=int, default=50,
                        help='Max concurrent threads for sync mode')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print dates without invoking')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Only aggregate results from S3 (no Lambda invocation)')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON string of config overrides')
    args = parser.parse_args()

    # Generate trading dates
    dates = generate_trading_dates(args.start, args.end)
    print(f"Date range: {args.start} to {args.end}")
    print(f"Trading days: {len(dates)}")

    if args.dry_run:
        print("\n[DRY RUN] Would invoke Lambda for:")
        for d in dates:
            print(f"  {d}")
        return

    config = json.loads(args.config) if args.config else None

    if args.aggregate_only:
        # Just download and combine results from S3
        combined_df = aggregate_results_from_s3(dates)
        if not combined_df.empty:
            out_path = f"/Users/sreenathreddy/Downloads/UniTrader-project/results/aggregated_{args.start}_to_{args.end}.csv"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            combined_df.to_csv(out_path, index=False)
            print(f"\nSaved aggregated results to: {out_path}")
        return

    start_time = time.time()

    if args.mode == 'sync':
        results = run_parallel_sync(dates, config, args.max_workers)
    else:
        results = run_async_all(dates, config)

    elapsed = time.time() - start_time
    print(f"\nTotal orchestration time: {elapsed:.1f}s")

    if args.mode == 'sync':
        print_final_summary(results)

        # Aggregate from S3
        combined_df = aggregate_results_from_s3(dates)
        if not combined_df.empty:
            out_dir = '/Users/sreenathreddy/Downloads/UniTrader-project/results'
            os.makedirs(out_dir, exist_ok=True)
            out_path = f"{out_dir}/aggregated_{args.start}_to_{args.end}.csv"
            combined_df.to_csv(out_path, index=False)
            print(f"\nSaved aggregated results to: {out_path}")

            # Save to S3 too
            csv_data = combined_df.to_csv(index=False)
            agg_key = f"{S3_PREFIX}/aggregated_{args.start}_to_{args.end}.csv"
            s3_client.put_object(Bucket=S3_BUCKET, Key=agg_key, Body=csv_data.encode('utf-8'))
            print(f"Saved to s3://{S3_BUCKET}/{agg_key}")
    else:
        print(f"\nAsync mode: Lambda functions are running independently.")
        print(f"Each day sends its own Telegram notification.")
        print(f"Use --aggregate-only to collect results when done:")
        print(f"  python orchestrator.py --start {args.start} --end {args.end} --aggregate-only")


if __name__ == '__main__':
    main()
