"""
First Target Hit Analyzer
=========================
Analyzes trades before first target hit, maximum loss before first target,
and number of trades to get first target for each day.
"""

import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
import json

# S3 Configuration
S3_BUCKET = 'tradelayout-backtest-dev'
S3_PREFIX = '5ema-results'
AWS_REGION = 'us-east-1'

def get_s3_client():
    """Get S3 client."""
    return boto3.client('s3', region_name=AWS_REGION)

def list_s3_files():
    """List all CSV files in S3 bucket."""
    s3_client = get_s3_client()
    files = []
    
    try:
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
        return sorted(files, key=lambda x: x['key'])
    except Exception as e:
        print(f"Error listing S3 files: {e}")
        return []

def download_and_analyze_file(file_info):
    """Download and analyze a single CSV file."""
    s3_client = get_s3_client()
    key = file_info['key']
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        
        if df.empty:
            return None
        
        # Extract date from key
        date_str = key.split('/')[-1].replace('.csv', '')
        
        # Sort by entry time to get chronological order
        df = df.sort_values('entry_time')
        
        # Find first target hit
        target_trades = df[df['exit_type'] == 'TARGET']
        
        if target_trades.empty:
            # No target hits for the day
            return {
                'date': date_str,
                'total_trades': len(df),
                'first_target_trade': None,
                'trades_before_first_target': len(df),
                'max_loss_before_first_target': df['cumulative_pnl'].min(),
                'first_target_pnl': None,
                'first_target_time': None,
                'first_target_hit': False,
                'day_pnl': df['trade_pnl'].sum(),
                'all_trades_before_target': df.to_dict('records')
            }
        
        # Get first target hit
        first_target = target_trades.iloc[0]
        first_target_index = first_target.name
        
        # Get trades before first target
        trades_before_target = df.loc[:first_target_index-1] if first_target_index > 0 else pd.DataFrame()
        
        # Calculate metrics
        trades_before_first_target = len(trades_before_target)
        max_loss_before_first_target = trades_before_target['cumulative_pnl'].min() if not trades_before_target.empty else 0
        
        return {
            'date': date_str,
            'total_trades': len(df),
            'first_target_trade': int(first_target_index) + 1,  # 1-based
            'trades_before_first_target': trades_before_first_target,
            'max_loss_before_first_target': max_loss_before_first_target,
            'first_target_pnl': first_target['trade_pnl'],
            'first_target_time': first_target['exit_time'],
            'first_target_hit': True,
            'day_pnl': df['trade_pnl'].sum(),
            'all_trades_before_target': trades_before_target.to_dict('records')
        }
        
    except Exception as e:
        print(f"  ❌ Error processing {key}: {e}")
        return {
            'date': key.split('/')[-1].replace('.csv', ''),
            'error': str(e),
            'first_target_hit': False
        }

def analyze_first_target_from_s3():
    """Analyze first target hits from S3 data."""
    print("🎯 Analyzing First Target Hits from S3...")
    print("="*60)
    
    # List all files
    files = list_s3_files()
    print(f"📁 Found {len(files)} CSV files")
    
    if not files:
        print("No files found!")
        return
    
    # Process files in parallel
    print(f"\n📥 Processing files...")
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(download_and_analyze_file, file): file['key']
            for file in files
        }
        
        for future in as_completed(futures):
            result = future.result()
            if result and 'error' not in result:
                results.append(result)
                status = "🎯" if result['first_target_hit'] else "❌"
                print(f"  {status} {result['date']}: {result['trades_before_first_target']} trades before first target")
    
    print(f"\n✅ Successfully processed {len(results)} files")
    return results

def analyze_first_target_from_local(csv_path):
    """Analyze first target hits from local CSV file."""
    print(f"🎯 Analyzing First Target Hits from Local CSV...")
    print(f"📁 File: {csv_path}")
    print("="*60)
    
    try:
        df = pd.read_csv(csv_path)
        
        # Group by date
        results = []
        
        for date, group in df.groupby('date'):
            # Sort by entry time
            group_sorted = group.sort_values('entry_time')
            
            # Find first target hit
            target_trades = group_sorted[group_sorted['exit_type'] == 'TARGET']
            
            if target_trades.empty:
                # No target hits for the day
                results.append({
                    'date': date,
                    'total_trades': len(group_sorted),
                    'first_target_trade': None,
                    'trades_before_first_target': len(group_sorted),
                    'max_loss_before_first_target': group_sorted['cumulative_pnl'].min(),
                    'first_target_pnl': None,
                    'first_target_time': None,
                    'first_target_hit': False,
                    'day_pnl': group_sorted['trade_pnl'].sum()
                })
            else:
                # Get first target hit
                first_target = target_trades.iloc[0]
                first_target_index = first_target.name
                
                # Get trades before first target
                trades_before_target = group_sorted.loc[:first_target_index-1] if first_target_index > 0 else pd.DataFrame()
                
                # Calculate metrics
                trades_before_first_target = len(trades_before_target)
                max_loss_before_first_target = trades_before_target['cumulative_pnl'].min() if not trades_before_target.empty else 0
                
                results.append({
                    'date': date,
                    'total_trades': len(group_sorted),
                    'first_target_trade': int(first_target_index) + 1,
                    'trades_before_first_target': trades_before_first_target,
                    'max_loss_before_first_target': max_loss_before_first_target,
                    'first_target_pnl': first_target['trade_pnl'],
                    'first_target_time': first_target['exit_time'],
                    'first_target_hit': True,
                    'day_pnl': group_sorted['trade_pnl'].sum()
                })
        
        print(f"✅ Successfully processed {len(results)} days")
        return results
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return []

def generate_first_target_summary(results):
    """Generate comprehensive summary of first target analysis."""
    if not results:
        print("❌ No data to analyze!")
        return
    
    df = pd.DataFrame(results)
    
    # Separate days with and without target hits
    with_target = df[df['first_target_hit'] == True]
    without_target = df[df['first_target_hit'] == False]
    
    print("="*80)
    print("  FIRST TARGET HIT ANALYSIS")
    print("="*80)
    
    print(f"\n📊 OVERALL SUMMARY")
    print(f"  Total Days Analyzed    : {len(df)}")
    print(f"  Days with Target Hit   : {len(with_target)} ({len(with_target)/len(df)*100:.1f}%)")
    print(f"  Days without Target    : {len(without_target)} ({len(without_target)/len(df)*100:.1f}%)")
    
    if not with_target.empty:
        print(f"\n🎯 DAYS WITH FIRST TARGET HIT")
        print(f"  Avg Trades Before 1st Target : {with_target['trades_before_first_target'].mean():.1f}")
        print(f"  Median Trades Before 1st Target : {with_target['trades_before_first_target'].median():.1f}")
        print(f"  Min Trades Before 1st Target  : {with_target['trades_before_first_target'].min()}")
        print(f"  Max Trades Before 1st Target  : {with_target['trades_before_first_target'].max()}")
        
        print(f"\n💰 MAX LOSS BEFORE FIRST TARGET")
        print(f"  Avg Max Loss Before 1st Target : {with_target['max_loss_before_first_target'].mean():+.2f} pts")
        print(f"  Worst Max Loss Before 1st Target : {with_target['max_loss_before_first_target'].min():+.2f} pts")
        print(f"  Best Max Loss Before 1st Target  : {with_target['max_loss_before_first_target'].max():+.2f} pts")
        
        print(f"\n⏱️  FIRST TARGET DETAILS")
        print(f"  Avg First Target Trade Number : {with_target['first_target_trade'].mean():.1f}")
        print(f"  Avg First Target P&L          : {with_target['first_target_pnl'].mean():+.2f} pts")
        
        # Distribution of trades before first target
        print(f"\n📈 DISTRIBUTION - TRADES BEFORE FIRST TARGET")
        distribution = with_target['trades_before_first_target'].value_counts().sort_index()
        for trades, count in distribution.head(10).items():
            print(f"  {trades} trades: {count} days ({count/len(with_target)*100:.1f}%)")
        
        # Days that hit target on first trade
        first_trade_targets = with_target[with_target['trades_before_first_target'] == 0]
        print(f"\n🚀 TARGET ON FIRST TRADE")
        print(f"  Days with target on 1st trade: {len(first_trade_targets)} ({len(first_trade_targets)/len(with_target)*100:.1f}%)")
        
        # Days that needed many trades before target
        many_trades = with_target[with_target['trades_before_first_target'] >= 20]
        print(f"  Days needing 20+ trades: {len(many_trades)} ({len(many_trades)/len(with_target)*100:.1f}%)")
    
    if not without_target.empty:
        print(f"\n❌ DAYS WITHOUT ANY TARGET HIT")
        print(f"  Total Days             : {len(without_target)}")
        print(f"  Avg Trades per Day     : {without_target['total_trades'].mean():.1f}")
        print(f"  Avg Daily P&L          : {without_target['day_pnl'].mean():+.2f} pts")
        print(f"  Worst Daily Loss        : {without_target['day_pnl'].min():+.2f} pts")
        print(f"  Best Daily P&L          : {without_target['day_pnl'].max():+.2f} pts")
    
    # Top 10 days with most trades before first target
    if not with_target.empty:
        print(f"\n🏆 TOP 10 DAYS - MOST TRADES BEFORE FIRST TARGET")
        most_trades = with_target.nlargest(10, 'trades_before_first_target')
        for _, row in most_trades.iterrows():
            print(f"  {row['date']}: {row['trades_before_first_target']} trades before target | Max loss: {row['max_loss_before_first_target']:+.2f} pts")
    
    # Top 10 days with worst max loss before first target
    if not with_target.empty:
        print(f"\n💔 TOP 10 DAYS - WORST MAX LOSS BEFORE FIRST TARGET")
        worst_loss = with_target.nsmallest(10, 'max_loss_before_first_target')
        for _, row in worst_loss.iterrows():
            print(f"  {row['date']}: Max loss {row['max_loss_before_first_target']:+.2f} pts | {row['trades_before_first_target']} trades before target")
    
    # Days with target on first trade
    if not with_target.empty:
        first_trade_targets = with_target[with_target['trades_before_first_target'] == 0]
        if not first_trade_targets.empty:
            print(f"\n🎯 DAYS WITH TARGET ON FIRST TRADE ({len(first_trade_targets)} days)")
            for _, row in first_trade_targets.head(10).iterrows():
                print(f"  {row['date']}: Target on 1st trade | P&L: {row['first_target_pnl']:+.2f} pts | Day P&L: {row['day_pnl']:+.2f} pts")
    
    # Save detailed results
    output_file = f'first_target_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_days': len(df),
        'days_with_target': len(with_target),
        'days_without_target': len(without_target),
        'target_hit_rate': len(with_target)/len(df)*100,
        'avg_trades_before_first_target': with_target['trades_before_first_target'].mean() if not with_target.empty else 0,
        'median_trades_before_first_target': with_target['trades_before_first_target'].median() if not with_target.empty else 0,
        'max_trades_before_first_target': with_target['trades_before_first_target'].max() if not with_target.empty else 0,
        'avg_max_loss_before_first_target': with_target['max_loss_before_first_target'].mean() if not with_target.empty else 0,
        'worst_max_loss_before_first_target': with_target['max_loss_before_first_target'].min() if not with_target.empty else 0,
        'days_with_target_on_first_trade': len(with_target[with_target['trades_before_first_target'] == 0]) if not with_target.empty else 0
    }
    
    summary_file = f'first_target_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"💾 Summary saved to: {summary_file}")
    
    print(f"\n{'='*80}\n")
    
    return df

def main():
    """Main analyzer function."""
    print("🎯 First Target Hit Analyzer")
    print("="*50)
    
    # Check AWS credentials
    if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
        print("✅ AWS credentials found - analyzing from S3")
        results = analyze_first_target_from_s3()
    else:
        print("⚠️  No AWS credentials - using local CSV file")
        csv_path = '/Users/sreenathreddy/Downloads/UniTrader-project/results/s3-aggregated_2024-06-01_to_2024-12-31.csv'
        if os.path.exists(csv_path):
            results = analyze_first_target_from_local(csv_path)
        else:
            print(f"❌ CSV file not found: {csv_path}")
            return
    
    # Generate summary
    if results:
        generate_first_target_summary(results)
    else:
        print("❌ No results to analyze")

if __name__ == '__main__':
    main()
