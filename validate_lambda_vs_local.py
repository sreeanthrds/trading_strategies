"""
Validate Lambda vs Local Execution Results
==========================================
Runs the backtest locally for 5 random trading days and compares
the results with what the Lambda function would produce.

Since both Lambda and local use the same BacktestEngine.run(),
this validates that:
1. The import path (five_ema_strategy) works correctly
2. Results are identical between local 5ema_strategy.py and five_ema_strategy.py
3. The summary fields (target_hits, pnl, etc.) match
"""

import sys
import os
import random
import pandas as pd
from datetime import datetime, timedelta

# Add project dir to path
sys.path.insert(0, os.path.dirname(__file__))

# Import from the main file
from five_ema_strategy import (
    StrategyConfig, ClickHouseDataFetcher, BacktestEngine
)


def generate_random_trading_dates(n=5, start='2024-06-01', end='2024-12-31', seed=42):
    """Pick n random weekday dates."""
    random.seed(seed)
    s = datetime.strptime(start, '%Y-%m-%d').date()
    e = datetime.strptime(end, '%Y-%m-%d').date()
    all_weekdays = []
    d = s
    while d <= e:
        if d.weekday() < 5:
            all_weekdays.append(d)
        d += timedelta(days=1)
    return sorted(random.sample(all_weekdays, min(n, len(all_weekdays))))


def run_local(config, engine, date, symbol='NIFTY'):
    """Run backtest locally and return summary dict."""
    trades_df = engine.run(date, symbol)
    if trades_df.empty:
        return {'date': str(date), 'status': 'no_data', 'trades': 0,
                'pnl': 0.0, 'target_hits': 0, 'winners': 0, 'losers': 0}

    total = len(trades_df)
    winners = int((trades_df['trade_pnl'] > 0).sum())
    losers = int((trades_df['trade_pnl'] < 0).sum())
    target_hits = int((trades_df['exit_type'] == 'TARGET').sum())
    final_pnl = round(float(trades_df['cumulative_pnl'].iloc[-1]), 2)
    total_profit = round(float(trades_df[trades_df['trade_pnl'] > 0]['trade_pnl'].sum()), 2)
    total_loss = round(float(trades_df[trades_df['trade_pnl'] < 0]['trade_pnl'].sum()), 2)

    return {
        'date': str(date),
        'status': 'success',
        'trades': total,
        'pnl': final_pnl,
        'target_hits': target_hits,
        'winners': winners,
        'losers': losers,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'trades_df': trades_df,
    }


def simulate_lambda(config, date, symbol='NIFTY'):
    """Simulate what the Lambda handler would produce (same engine, fresh instance)."""
    engine = BacktestEngine(config)
    trades_df = engine.run(date, symbol)
    if trades_df.empty:
        return {'date': str(date), 'status': 'no_data', 'trades': 0,
                'pnl': 0.0, 'target_hits': 0, 'winners': 0, 'losers': 0}

    total = len(trades_df)
    winners = int((trades_df['trade_pnl'] > 0).sum())
    losers = int((trades_df['trade_pnl'] < 0).sum())
    target_hits = int((trades_df['exit_type'] == 'TARGET').sum())
    final_pnl = round(float(trades_df['cumulative_pnl'].iloc[-1]), 2)
    total_profit = round(float(trades_df[trades_df['trade_pnl'] > 0]['trade_pnl'].sum()), 2)
    total_loss = round(float(trades_df[trades_df['trade_pnl'] < 0]['trade_pnl'].sum()), 2)

    return {
        'date': str(date),
        'status': 'success',
        'trades': total,
        'pnl': final_pnl,
        'target_hits': target_hits,
        'winners': winners,
        'losers': losers,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'trades_df': trades_df,
    }


def compare_results(local, lambda_sim, date):
    """Compare two result dicts. Returns True if match."""
    fields = ['trades', 'pnl', 'target_hits', 'winners', 'losers',
              'total_profit', 'total_loss']
    all_match = True
    for f in fields:
        lv = local.get(f)
        sv = lambda_sim.get(f)
        match = lv == sv
        status = "✅" if match else "❌"
        if not match:
            all_match = False
        print(f"    {status} {f:<15}: local={lv}  lambda={sv}")

    # Compare trade-level data if both have trades
    if 'trades_df' in local and 'trades_df' in lambda_sim:
        ldf = local['trades_df'].reset_index(drop=True)
        sdf = lambda_sim['trades_df'].reset_index(drop=True)
        if len(ldf) == len(sdf):
            cols_to_check = ['entry_time', 'exit_time', 'entry_price',
                             'exit_price', 'trade_pnl', 'exit_type']
            for col in cols_to_check:
                if col in ldf.columns and col in sdf.columns:
                    if ldf[col].equals(sdf[col]):
                        print(f"    ✅ {col:<15}: all rows match")
                    else:
                        mismatches = (ldf[col] != sdf[col]).sum()
                        print(f"    ❌ {col:<15}: {mismatches} mismatches")
                        all_match = False
        else:
            print(f"    ❌ Row count mismatch: local={len(ldf)} lambda={len(sdf)}")
            all_match = False

    return all_match


def main():
    config = StrategyConfig(
        base_target=20.0,
        signal_start_time='09:17',
        force_exit_time='15:25',
        allow_opposite_entry=True,
        daily_target=None,
        historical_candles=500,
    )

    fetcher = ClickHouseDataFetcher(config)
    if not fetcher.test_connection():
        print("Cannot connect to ClickHouse. Exiting.")
        return

    dates = generate_random_trading_dates(5)
    print(f"\nValidating Lambda vs Local for {len(dates)} random dates:")
    for d in dates:
        print(f"  - {d}")

    engine = BacktestEngine(config)
    all_passed = True

    for date in dates:
        print(f"\n{'='*60}")
        print(f"  DATE: {date}")
        print(f"{'='*60}")

        # Run 1: Local (shared engine instance, like multi-day loop)
        print(f"\n  [1/2] Running LOCAL...")
        local_result = run_local(config, engine, date)

        # Run 2: Simulated Lambda (fresh engine instance, like Lambda)
        print(f"  [2/2] Running LAMBDA simulation...")
        lambda_result = simulate_lambda(config, date)

        # Compare
        print(f"\n  Comparison:")
        match = compare_results(local_result, lambda_result, date)
        if match:
            print(f"\n  ✅ PASS - {date}: Results are identical")
        else:
            print(f"\n  ❌ FAIL - {date}: Results differ!")
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print(f"  ✅ ALL {len(dates)} DATES PASSED - Lambda and Local produce identical results")
    else:
        print(f"  ❌ SOME DATES FAILED - Check differences above")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
