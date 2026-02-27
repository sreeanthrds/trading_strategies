#!/usr/bin/env python3
"""
Run Backtest with External Configuration
======================================
Usage:
  # Using environment variables
  export STRATEGY_BASE_TARGET=25.0
  export STRATEGY_SIGNAL_START_TIME=09:20
  export STRATEGY_DAILY_TARGET=100.0
  python run_with_config.py --date 2024-10-03

  # Using JSON config file
  python run_with_config.py --date 2024-10-03 --config my_config.json

  # Override specific parameters
  python run_with_config.py --date 2024-10-03 --base-target 30.0 --daily-target 50.0
"""

import argparse
import os
import sys
from datetime import date, datetime
from config_loader import load_config_from_env, load_config_from_json, create_strategy_config, print_config_summary

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run backtest with external configuration')
    
    # Date selection
    parser.add_argument('--date', type=str, help='Date to run (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    # Config sources
    parser.add_argument('--config', type=str, help='JSON config file path')
    parser.add_argument('--env', action='store_true', help='Load from environment variables')
    
    # Direct overrides
    parser.add_argument('--base-target', type=float, help='Base target points')
    parser.add_argument('--signal-start-time', type=str, help='Signal start time (HH:MM)')
    parser.add_argument('--force-exit-time', type=str, help='Force exit time (HH:MM)')
    parser.add_argument('--daily-target', type=float, help='Daily target points')
    parser.add_argument('--historical-candles', type=int, help='Historical candles for warmup')
    parser.add_argument('--no-opposite-entry', action='store_true', help='Disable opposite entry')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()

def create_external_config(args) -> 'ExternalConfig':
    """Create external config from args."""
    from config_loader import ExternalConfig
    
    # Load from file or environment
    if args.config:
        external = load_config_from_json(args.config)
    elif args.env:
        external = load_config_from_env()
    else:
        external = ExternalConfig()
    
    # Apply command line overrides
    if args.base_target is not None:
        external.base_target = args.base_target
    if args.signal_start_time is not None:
        external.signal_start_time = args.signal_start_time
    if args.force_exit_time is not None:
        external.force_exit_time = args.force_exit_time
    if args.daily_target is not None:
        external.daily_target = args.daily_target
    if args.historical_candles is not None:
        external.historical_candles = args.historical_candles
    if args.no_opposite_entry:
        external.allow_opposite_entry = False
    
    return external

def run_single_date(config, run_date, output_dir):
    """Run backtest for a single date."""
    from five_ema_strategy import BacktestEngine, ClickHouseDataFetcher
    
    print(f"\nRunning backtest for {run_date}")
    print(f"{'-'*50}")
    
    # Test connection
    fetcher = ClickHouseDataFetcher(config)
    if not fetcher.test_connection():
        print("Cannot connect to ClickHouse. Exiting.")
        return None
    
    # Run backtest
    engine = BacktestEngine(config)
    try:
        trades_df = engine.run(run_date)
        if trades_df.empty:
            print(f"No trades on {run_date}")
            return None
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        csv_path = f"{output_dir}/trades_{run_date}.csv"
        trades_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        return trades_df
    except Exception as e:
        print(f"Error on {run_date}: {e}")
        return None

def run_date_range(config, start_date, end_date, output_dir):
    """Run backtest for a date range."""
    from five_ema_strategy import BacktestEngine, ClickHouseDataFetcher
    from datetime import timedelta
    
    print(f"\nRunning backtest from {start_date} to {end_date}")
    print(f"{'-'*50}")
    
    # Test connection
    fetcher = ClickHouseDataFetcher(config)
    if not fetcher.test_connection():
        print("Cannot connect to ClickHouse. Exiting.")
        return None
    
    # Collect all trades
    engine = BacktestEngine(config)
    all_trades = []
    current_date = start_date
    total_days = 0
    successful_days = 0
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() < 5:
            total_days += 1
            print(f"\nProcessing {current_date} ({total_days} trading days)...")
            
            try:
                daily_trades = engine.run(current_date)
                if not daily_trades.empty:
                    all_trades.append(daily_trades)
                    successful_days += 1
                    print(f"  ✓ {len(daily_trades)} trades")
                else:
                    print(f"  - No trades")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        current_date += timedelta(days=1)
    
    # Combine and save results
    if all_trades:
        import pandas as pd
        trades_df = pd.concat(all_trades, ignore_index=True)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        csv_path = f"{output_dir}/trades_{start_date}_to_{end_date}.csv"
        trades_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        # Print summary
        from five_ema_strategy import ReportGenerator
        ReportGenerator.print_summary(trades_df)
        
        return trades_df
    else:
        print("\nNo trades executed in the entire period.")
        return None

def main():
    """Main function."""
    args = parse_args()
    
    # Create configuration
    external_config = create_external_config(args)
    config = create_strategy_config(external_config)
    
    # Print configuration
    print_config_summary(config)
    
    # Determine date range
    if args.date:
        run_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        run_single_date(config, run_date, args.output_dir)
    elif args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
        run_date_range(config, start_date, end_date, args.output_dir)
    else:
        # Default to today
        today = date.today()
        run_single_date(config, today, args.output_dir)

if __name__ == '__main__':
    main()
