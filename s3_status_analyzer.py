"""
S3 Status Analyzer
==================
Analyze S3 backtest status and provide comprehensive summary.
Works with or without AWS credentials.
"""

import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

# S3 Configuration
S3_BUCKET = 'tradelayout-backtest-dev'
S3_PREFIX = '5ema-results'
AWS_REGION = 'us-east-1'

def check_aws_credentials():
    """Check if AWS credentials are available."""
    return os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY')

def get_s3_client():
    """Get S3 client if credentials are available."""
    if check_aws_credentials():
        return boto3.client('s3', region_name=AWS_REGION)
    else:
        return None

def list_s3_files_with_credentials():
    """List all files in S3 using credentials."""
    s3_client = get_s3_client()
    if not s3_client:
        return None
    
    print(f"📁 Listing files in s3://{S3_BUCKET}/{S3_PREFIX}/")
    
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    try:
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
            for obj in page.get('Contents', []):
                key = obj['Key']
                files.append({
                    'key': key,
                    'size': obj['Size'],
                    'last_modified': obj['LastModified']
                })
        
        print(f"  Found {len(files)} files")
        return sorted(files, key=lambda x: x['key'])
    
    except Exception as e:
        print(f"  ❌ Error listing S3 files: {e}")
        return None

def analyze_from_local_csv(csv_path):
    """Analyze from existing local CSV file."""
    print(f"📊 Analyzing from local file: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Daily aggregation
        daily = df.groupby('date').agg(
            trades=('trade_pnl', 'count'),
            target_hits=('exit_type', lambda x: (x == 'TARGET').sum()),
            force_exits=('exit_type', lambda x: (x == 'FORCE_EXIT').sum()),
            stop_losses=('exit_type', lambda x: (x == 'STOP_LOSS').sum()),
            winners=('trade_pnl', lambda x: (x > 0).sum()),
            losers=('trade_pnl', lambda x: (x < 0).sum()),
            daily_pnl=('trade_pnl', 'sum'),
            total_profit=('trade_pnl', lambda x: x[x > 0].sum() if (x > 0).any() else 0),
            total_loss=('trade_pnl', lambda x: x[x < 0].sum() if (x < 0).any() else 0),
            avg_win=('trade_pnl', lambda x: x[x > 0].mean() if (x > 0).any() else 0),
            avg_loss=('trade_pnl', lambda x: x[x < 0].mean() if (x < 0).any() else 0),
            max_win=('trade_pnl', 'max'),
            max_loss=('trade_pnl', 'min'),
            avg_hold_time=('hold_duration_seconds', 'mean'),
            max_drawdown=('cumulative_pnl', 'min'),
            peak_pnl=('cumulative_pnl', 'max'),
            avg_mfe=('mfe_points', 'mean'),
            avg_mae=('mae_points', 'mean'),
            avg_r_multiple=('r_multiple', 'mean'),
            avg_target_reach=('target_reached_pct', 'mean'),
        ).reset_index()
        
        return daily, df
    
    except Exception as e:
        print(f"  ❌ Error reading CSV: {e}")
        return None, None

def analyze_from_s3():
    """Analyze directly from S3."""
    files = list_s3_files_with_credentials()
    if not files:
        return None, None
    
    s3_client = get_s3_client()
    all_trades = []
    daily_summaries = []
    
    print(f"\n📥 Downloading and analyzing {len(files)} files...")
    
    def download_and_analyze(file_info):
        key = file_info['key']
        if not key.endswith('.csv') or key.endswith('_summary.csv'):
            return None
        
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            csv_content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_content))
            
            if df.empty:
                return None
            
            # Extract date from key
            date_str = key.split('/')[-1].replace('.csv', '')
            
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
            
            return daily_summary, df
            
        except Exception as e:
            print(f"  ❌ Error processing {key}: {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_and_analyze, file) for file in files]
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                daily_summary, df = result
                daily_summaries.append(daily_summary)
                all_trades.append(df)
                print(f"  ✅ {daily_summary['date']}: {daily_summary['trades']} trades")
    
    if daily_summaries:
        daily_df = pd.DataFrame(daily_summaries)
        daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
        trades_df = pd.concat(all_trades, ignore_index=True)
        return daily_df, trades_df
    
    return None, None

def generate_comprehensive_summary(daily_df, trades_df, source="Local CSV"):
    """Generate comprehensive summary like the detailed CSV format."""
    if daily_df is None:
        print("❌ No data to analyze!")
        return
    
    # Sort by date
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
    print("="*80)
    print(f"  COMPREHENSIVE 5EMA STRATEGY SUMMARY ({source})")
    print("="*80)
    
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
    output_path = f's3_status_detailed_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    daily_df.to_csv(output_path, index=False)
    print(f"\n💾 Detailed daily summary saved to: {output_path}")
    
    print(f"\n{'='*80}\n")
    
    return daily_df, trades_df

def main():
    """Main analyzer function."""
    print("🔍 S3 Trade Status Analyzer")
    print("="*50)
    
    # Check if we have AWS credentials
    if check_aws_credentials():
        print("✅ AWS credentials found - analyzing directly from S3")
        daily_df, trades_df = analyze_from_s3()
        if daily_df is not None:
            generate_comprehensive_summary(daily_df, trades_df, "S3 Direct")
        else:
            print("❌ Failed to analyze from S3")
    else:
        print("⚠️  No AWS credentials - using local CSV file")
        
        # Try to find the aggregated CSV file
        csv_path = '/Users/sreenathreddy/Downloads/UniTrader-project/results/s3-aggregated_2024-06-01_to_2024-12-31.csv'
        if os.path.exists(csv_path):
            daily_df, trades_df = analyze_from_local_csv(csv_path)
            if daily_df is not None:
                generate_comprehensive_summary(daily_df, trades_df, "Local CSV")
            else:
                print("❌ Failed to analyze local CSV")
        else:
            print(f"❌ CSV file not found: {csv_path}")
            print("Please provide the correct path to the aggregated CSV file")
    
    # Also show S3 status if possible
    s3_client = get_s3_client()
    if s3_client:
        print("\n🔍 Checking S3 bucket status...")
        try:
            # List files
            files = list_s3_files_with_credentials()
            if files:
                print(f"\n📁 S3 Bucket Contents (s3://{S3_BUCKET}/{S3_PREFIX}/)")
                print(f"{'File':<30} {'Size':>10} {'Last Modified':<20}")
                print(f"{'-'*60}")
                
                for file in files[:20]:  # Show first 20 files
                    key = file['key']
                    size = file['size']
                    modified = file['last_modified'].strftime('%Y-%m-%d %H:%M:%S')
                    filename = key.split('/')[-1]
                    print(f"{filename:<30} {size:>10} {modified:<20}")
                
                if len(files) > 20:
                    print(f"... and {len(files) - 20} more files")
        except Exception as e:
            print(f"❌ Error checking S3: {e}")

if __name__ == '__main__':
    main()
