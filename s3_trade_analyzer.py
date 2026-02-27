"""
S3 Trade Analyzer
=================
Analyze trades directly from S3 bucket and generate comprehensive summary.
"""

import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from io import StringIO

# S3 Configuration
S3_BUCKET = 'tradelayout-backtest-dev'
S3_PREFIX = '5ema-results'
AWS_REGION = 'us-east-1'

# Initialize S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION)

def list_s3_files():
    """List all CSV files in S3 bucket."""
    print(f"📁 Listing files in s3://{S3_BUCKET}/{S3_PREFIX}/")
    
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.csv') and not key.endswith('_summary.csv'):
                files.append({
                    'key': key,
                    'size': obj['Size'],
                    'last_modified': obj['LastModified']
                })
    
    print(f"  Found {len(files)} CSV files")
    return sorted(files, key=lambda x: x['key'])

def download_csv_from_s3(key):
    """Download and parse CSV file from S3."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        return key, df
    except Exception as e:
        print(f"  ❌ Error downloading {key}: {e}")
        return key, None

def analyze_trades_from_s3():
    """Analyze all trades from S3 and generate comprehensive summary."""
    print("="*80)
    print("  S3 TRADE ANALYSIS FOR 5EMA STRATEGY")
    print("="*80)
    
    # List all files
    files = list_s3_files()
    
    if not files:
        print("No CSV files found in S3 bucket!")
        return
    
    # Download and analyze files
    print(f"\n📥 Downloading and analyzing {len(files)} files...")
    
    all_trades = []
    daily_summaries = []
    successful_downloads = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all download tasks
        future_to_key = {
            executor.submit(download_csv_from_s3, file['key']): file['key']
            for file in files
        }
        
        # Process completed downloads
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                file_key, df = future.result()
                
                if df is not None and not df.empty:
                    all_trades.append(df)
                    successful_downloads += 1
                    
                    # Extract date from key (format: 5ema-results/YYYY-MM-DD.csv)
                    date_str = file_key.split('/')[-1].replace('.csv', '')
                    
                    # Calculate daily summary
                    daily_summary = {
                        'date': date_str,
                        'trades': len(df),
                        'target_hits': len(df[df['exit_type'] == 'TARGET']),
                        'force_exits': len(df[df['exit_type'] == 'FORCE_EXIT']),
                        'stop_losses': len(df[df['exit_type'] == 'STOP_LOSS']),
                        'winners': len(df[df['trade_pnl'] > 0]),
                        'losers': len(df[df['trade_pnl'] < 0]),
                        'daily_pnl': df['trade_pnl'].sum(),
                        'total_profit': df[df['trade_pnl'] > 0]['trade_pnl'].sum() if (df['trade_pnl'] > 0).any() else 0,
                        'total_loss': df[df['trade_pnl'] < 0]['trade_pnl'].sum() if (df['trade_pnl'] < 0).any() else 0,
                        'avg_win': df[df['trade_pnl'] > 0]['trade_pnl'].mean() if (df['trade_pnl'] > 0).any() else 0,
                        'avg_loss': df[df['trade_pnl'] < 0]['trade_pnl'].mean() if (df['trade_pnl'] < 0).any() else 0,
                        'max_win': df['trade_pnl'].max(),
                        'max_loss': df['trade_pnl'].min(),
                        'avg_hold_time': df['hold_duration_seconds'].mean(),
                        'max_drawdown': df['cumulative_pnl'].min(),
                        'peak_pnl': df['cumulative_pnl'].max(),
                        'avg_mfe': df['mfe_points'].mean(),
                        'avg_mae': df['mae_points'].mean(),
                        'avg_r_multiple': df['r_multiple'].mean(),
                        'avg_target_reach': df['target_reached_pct'].mean(),
                    }
                    daily_summaries.append(daily_summary)
                    
                    print(f"  ✅ {date_str}: {len(df)} trades")
                else:
                    print(f"  ⚠️  {key}: No data")
                    
            except Exception as e:
                print(f"  ❌ Error processing {key}: {e}")
    
    print(f"\n📊 Successfully processed {successful_downloads}/{len(files)} files")
    
    if not all_trades:
        print("No trade data found!")
        return
    
    # Combine all trades
    trades_df = pd.concat(all_trades, ignore_index=True)
    daily_df = pd.DataFrame(daily_summaries)
    
    # Convert date column
    daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
    daily_df = daily_df.sort_values('date')
    
    # Calculate additional metrics
    daily_df['win_rate'] = (daily_df['winners'] / daily_df['trades'] * 100).round(2)
    daily_df['target_hit_rate'] = (daily_df['target_hits'] / daily_df['trades'] * 100).round(2)
    daily_df['profit_factor'] = (daily_df['total_profit'] / abs(daily_df['total_loss'])).round(2).replace([np.inf, -np.inf], 0)
    daily_df['avg_trade'] = (daily_df['daily_pnl'] / daily_df['trades']).round(2)
    daily_df['risk_reward'] = (abs(daily_df['avg_win'] / daily_df['avg_loss'])).round(2).replace([np.inf, -np.inf], 0)
    daily_df['month'] = pd.to_datetime(daily_df['date']).dt.to_period('M')
    
    # Monthly aggregation
    monthly = daily_df.groupby('month').agg(
        days=('date', 'count'),
        trades=('trades', 'sum'),
        target_hits=('target_hits', 'sum'),
        daily_pnl=('daily_pnl', 'sum'),
        winners=('winners', 'sum'),
        losers=('losers', 'sum'),
        profitable_days=('daily_pnl', lambda x: (x > 0).sum()),
        losing_days=('daily_pnl', lambda x: (x < 0).sum()),
    ).reset_index()
    monthly['monthly_pnl'] = monthly['daily_pnl']
    monthly['win_rate'] = (monthly['winners'] / monthly['trades'] * 100).round(2)
    monthly['avg_daily_pnl'] = (monthly['daily_pnl'] / monthly['days']).round(2)
    
    # Overall statistics
    total_days = len(daily_df)
    profitable_days = (daily_df['daily_pnl'] > 0).sum()
    losing_days = (daily_df['daily_pnl'] < 0).sum()
    flat_days = (daily_df['daily_pnl'] == 0).sum()
    
    # Print comprehensive summary
    print(f"\n📊 OVERALL PERFORMANCE")
    print(f"  Total Trading Days     : {total_days}")
    print(f"  Profitable Days        : {profitable_days} ({profitable_days/total_days*100:.1f}%)")
    print(f"  Losing Days            : {losing_days} ({losing_days/total_days*100:.1f}%)")
    print(f"  Flat Days              : {flat_days} ({flat_days/total_days*100:.1f}%)")
    print(f"  Total Trades           : {daily_df['trades'].sum()}")
    print(f"  Total Target Hits      : {daily_df['target_hits'].sum()}")
    print(f"  Total P&L              : {daily_df['daily_pnl'].sum():+,.2f} pts")
    print(f"  Total Profit           : {daily_df['total_profit'].sum():+,.2f} pts")
    print(f"  Total Loss             : {daily_df['total_loss'].sum():+,.2f} pts")
    print(f"  Overall Win Rate       : {(daily_df['winners'].sum()/daily_df['trades'].sum()*100):.1f}%")
    print(f"  Target Hit Rate        : {(daily_df['target_hits'].sum()/daily_df['trades'].sum()*100):.1f}%")
    print(f"  Profit Factor          : {(daily_df['total_profit'].sum()/abs(daily_df['total_loss'].sum())):.2f}")
    print(f"  Average Daily P&L      : {daily_df['daily_pnl'].mean():+.2f} pts")
    print(f"  Best Day               : {daily_df.loc[daily_df['daily_pnl'].idxmax(), 'daily_pnl']:+,.2f} pts")
    print(f"  Worst Day              : {daily_df.loc[daily_df['daily_pnl'].idxmin(), 'daily_pnl']:+,.2f} pts")
    
    print(f"\n📈 MONTHLY BREAKDOWN")
    print(f"{'Month':<10} {'Days':>5} {'Trades':>7} {'Targets':>8} {'P&L':>12} {'Win%':>6} {'Avg/P&L':>9}")
    print(f"{'-'*65}")
    for _, row in monthly.iterrows():
        month_str = str(row['month'])
        pnl_str = f"{row['monthly_pnl']:+,.0f}"
        avg_str = f"{row['avg_daily_pnl']:+,.0f}"
        print(f"{month_str:<10} {row['days']:>5} {row['trades']:>7} {row['target_hits']:>8} {pnl_str:>12} {row['win_rate']:>6.1f}% {avg_str:>9}")
    
    print(f"\n🎯 TARGET HIT ANALYSIS")
    target_dist = daily_df['target_hits'].value_counts().sort_index()
    print(f"  Target Hits Range     : {daily_df['target_hits'].min()} - {daily_df['target_hits'].max()}")
    print(f"  Average Target Hits   : {daily_df['target_hits'].mean():.1f} per day")
    print(f"  Days with 0 Targets   : {(daily_df['target_hits'] == 0).sum()} ({(daily_df['target_hits'] == 0).sum()/total_days*100:.1f}%)")
    print(f"  Days with 10+ Targets : {(daily_df['target_hits'] >= 10).sum()} ({(daily_df['target_hits'] >= 10).sum()/total_days*100:.1f}%)")
    
    print(f"\n📋 TOP 10 BEST DAYS")
    best_days = daily_df.nlargest(10, 'daily_pnl')[['date', 'daily_pnl', 'trades', 'target_hits', 'win_rate']]
    for _, row in best_days.iterrows():
        print(f"  {row['date']}: P&L {row['daily_pnl']:+,.2f} | {row['trades']} trades | {row['target_hits']} targets | {row['win_rate']:.1f}% win")
    
    print(f"\n📉 TOP 10 WORST DAYS")
    worst_days = daily_df.nsmallest(10, 'daily_pnl')[['date', 'daily_pnl', 'trades', 'target_hits', 'win_rate']]
    for _, row in worst_days.iterrows():
        print(f"  {row['date']}: P&L {row['daily_pnl']:+,.2f} | {row['trades']} trades | {row['target_hits']} targets | {row['win_rate']:.1f}% win")
    
    print(f"\n🏆 PERFORMANCE MILESTONES")
    print(f"  Highest Single Day P&L: {daily_df['daily_pnl'].max():+,.2f} pts on {daily_df.loc[daily_df['daily_pnl'].idxmax(), 'date']}")
    print(f"  Most Trades in a Day  : {daily_df['trades'].max()} trades on {daily_df.loc[daily_df['trades'].idxmax(), 'date']}")
    print(f"  Most Target Hits      : {daily_df['target_hits'].max()} targets on {daily_df.loc[daily_df['target_hits'].idxmax(), 'date']}")
    print(f"  Highest Win Rate       : {daily_df['win_rate'].max():.1f}% on {daily_df.loc[daily_df['win_rate'].idxmax(), 'date']}")
    print(f"  Best Profit Factor    : {daily_df['profit_factor'].max():.2f} on {daily_df.loc[daily_df['profit_factor'].idxmax(), 'date']}")
    
    # Streak analysis
    daily_df['is_profitable'] = daily_df['daily_pnl'] > 0
    daily_df['streak'] = (daily_df['is_profitable'] != daily_df['is_profitable'].shift()).cumsum()
    streaks = daily_df.groupby(['streak', 'is_profitable']).size().unstack(fill_value=0)
    
    if True in streaks.columns:
        max_win_streak = streaks[True].max()
        avg_win_streak = streaks[True][streaks[True] > 0].mean()
    else:
        max_win_streak = avg_win_streak = 0
    
    if False in streaks.columns:
        max_loss_streak = streaks[False].max()
        avg_loss_streak = streaks[False][streaks[False] > 0].mean()
    else:
        max_loss_streak = avg_loss_streak = 0
    
    print(f"\n🔄 STREAK ANALYSIS")
    print(f"  Longest Winning Streak: {max_win_streak} days")
    print(f"  Longest Losing Streak : {max_loss_streak} days")
    print(f"  Average Winning Streak: {avg_win_streak:.1f} days")
    print(f"  Average Losing Streak : {avg_loss_streak:.1f} days")
    
    # Risk analysis
    print(f"\n⚠️  RISK ANALYSIS")
    print(f"  Max Daily Drawdown    : {daily_df['max_drawdown'].min():+,.2f} pts")
    print(f"  Average Daily Range   : {daily_df['avg_mfe'].mean() + abs(daily_df['avg_mae'].mean()):.1f} pts")
    print(f"  Average Risk/Reward   : {daily_df['risk_reward'].mean():.2f}")
    print(f"  Average R-Multiple    : {daily_df['avg_r_multiple'].mean():.3f}")
    
    # Save detailed CSV
    output_path = 's3_detailed_summary.csv'
    daily_df.to_csv(output_path, index=False)
    print(f"\n💾 Detailed daily summary saved to: {output_path}")
    
    # Save all trades CSV
    all_trades_path = 's3_all_trades.csv'
    trades_df.to_csv(all_trades_path, index=False)
    print(f"💾 All trades saved to: {all_trades_path}")
    
    print(f"\n{'='*80}\n")
    
    return daily_df, trades_df

if __name__ == '__main__':
    # Check for AWS credentials
    if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
        print("❌ AWS credentials not found!")
        print("Please set:")
        print("  export AWS_ACCESS_KEY_ID='your_key'")
        print("  export AWS_SECRET_ACCESS_KEY='your_secret'")
        print("  export AWS_REGION='us-east-1'")
        exit(1)
    
    daily_df, trades_df = analyze_trades_from_s3()
