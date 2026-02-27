"""
Fix API Status Issues
=====================
Script to fix the FastAPI status endpoint and provide proper S3 analysis.
"""

import boto3
import pandas as pd
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

# S3 Configuration
S3_BUCKET = 'tradelayout-backtest-dev'
S3_PREFIX = '5ema-results'
AWS_REGION = 'us-east-1'

def get_s3_client():
    """Get S3 client."""
    return boto3.client('s3', region_name=AWS_REGION)

def list_s3_files():
    """List all files in S3 bucket."""
    s3_client = get_s3_client()
    files = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
            for obj in page.get('Contents', []):
                key = obj['Key']
                files.append({
                    'key': key,
                    'size': obj['Size'],
                    'last_modified': obj['LastModified']
                })
        return sorted(files, key=lambda x: x['key'])
    except Exception as e:
        print(f"Error listing S3 files: {e}")
        return []

def get_file_summary(key):
    """Get summary of a single CSV file from S3."""
    s3_client = get_s3_client()
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        
        if df.empty:
            return {
                'date': key.split('/')[-1].replace('.csv', ''),
                'status': 'no_data',
                'trades': 0,
                'pnl': 0.0,
                'winners': 0,
                'losers': 0,
                'target_hits': 0,
                'force_exits': 0,
                'error': None
            }
        
        # Calculate summary
        winners = len(df[df['trade_pnl'] > 0])
        losers = len(df[df['trade_pnl'] < 0])
        target_hits = len(df[df['exit_type'] == 'TARGET'])
        force_exits = len(df[df['exit_type'] == 'FORCE_EXIT'])
        
        return {
            'date': key.split('/')[-1].replace('.csv', ''),
            'status': 'success',
            'trades': len(df),
            'pnl': df['trade_pnl'].sum(),
            'winners': winners,
            'losers': losers,
            'target_hits': target_hits,
            'force_exits': force_exits,
            'error': None
        }
        
    except Exception as e:
        return {
            'date': key.split('/')[-1].replace('.csv', ''),
            'status': 'error',
            'trades': 0,
            'pnl': 0.0,
            'winners': 0,
            'losers': 0,
            'target_hits': 0,
            'force_exits': 0,
            'error': str(e)
        }

def analyze_s3_status():
    """Analyze S3 status and return comprehensive summary."""
    print("🔍 Analyzing S3 Status...")
    
    # List all files
    files = list_s3_files()
    csv_files = [f for f in files if f['key'].endswith('.csv') and not f['key'].endswith('_summary.csv')]
    
    print(f"  Found {len(csv_files)} CSV files")
    
    if not csv_files:
        return {
            'total_files': 0,
            'total_days': 0,
            'successful_days': 0,
            'total_trades': 0,
            'total_pnl': 0.0,
            'target_hits': 0,
            'files': []
        }
    
    # Process files in parallel
    summaries = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(get_file_summary, file['key']): file['key']
            for file in csv_files
        }
        
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            
            if summary['status'] == 'success':
                print(f"  ✅ {summary['date']}: {summary['trades']} trades, P&L {summary['pnl']:+.2f}")
            else:
                print(f"  ❌ {summary['date']}: {summary['status']}")
    
    # Calculate overall summary
    successful_days = sum(1 for s in summaries if s['status'] == 'success')
    total_trades = sum(s['trades'] for s in summaries)
    total_pnl = sum(s['pnl'] for s in summaries)
    target_hits = sum(s['target_hits'] for s in summaries)
    
    overall_summary = {
        'total_files': len(csv_files),
        'total_days': len(summaries),
        'successful_days': successful_days,
        'failed_days': len(summaries) - successful_days,
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'target_hits': target_hits,
        'files': summaries
    }
    
    return overall_summary

def generate_status_json():
    """Generate status JSON for API."""
    summary = analyze_s3_status()
    
    # Create status report
    status_report = {
        'timestamp': datetime.now().isoformat(),
        's3_bucket': S3_BUCKET,
        's3_prefix': S3_PREFIX,
        'summary': {
            'total_days': summary['total_days'],
            'successful_days': summary['successful_days'],
            'failed_days': summary['failed_days'],
            'success_rate': (summary['successful_days'] / summary['total_days'] * 100) if summary['total_days'] > 0 else 0,
            'total_trades': summary['total_trades'],
            'total_pnl': summary['total_pnl'],
            'target_hits': summary['target_hits'],
            'target_hit_rate': (summary['target_hits'] / summary['total_trades'] * 100) if summary['total_trades'] > 0 else 0,
            'avg_daily_pnl': summary['total_pnl'] / summary['successful_days'] if summary['successful_days'] > 0 else 0
        },
        'files': summary['files'][:50]  # First 50 files for preview
    }
    
    # Save status JSON
    status_file = f's3_status_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(status_file, 'w') as f:
        json.dump(status_report, f, indent=2, default=str)
    
    print(f"\n💾 Status saved to: {status_file}")
    
    return status_report

def print_detailed_status():
    """Print detailed status report."""
    summary = analyze_s3_status()
    
    print("\n" + "="*80)
    print("  S3 TRADE STATUS REPORT")
    print("="*80)
    
    print(f"\n📊 OVERALL STATUS")
    print(f"  Total Files           : {summary['total_files']}")
    print(f"  Total Days           : {summary['total_days']}")
    print(f"  Successful Days      : {summary['successful_days']}")
    print(f"  Failed Days          : {summary['failed_days']}")
    print(f"  Success Rate         : {(summary['successful_days']/summary['total_days']*100):.1f}%")
    print(f"  Total Trades         : {summary['total_trades']:,}")
    print(f"  Total P&L            : {summary['total_pnl']:+,.2f} pts")
    print(f"  Total Target Hits    : {summary['target_hits']}")
    print(f"  Target Hit Rate      : {(summary['target_hits']/summary['total_trades']*100):.1f}%")
    
    if summary['successful_days'] > 0:
        print(f"  Average Daily P&L    : {summary['total_pnl']/summary['successful_days']:+,.2f} pts")
        print(f"  Average Trades/Day   : {summary['total_trades']/summary['successful_days']:.1f}")
    
    # Show failed files
    failed_files = [f for f in summary['files'] if f['status'] != 'success']
    if failed_files:
        print(f"\n❌ FAILED FILES ({len(failed_files)})")
        for f in failed_files[:10]:
            print(f"  {f['date']}: {f['status']} - {f.get('error', 'Unknown error')}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more failed files")
    
    # Show best and worst days
    successful_files = [f for f in summary['files'] if f['status'] == 'success']
    if successful_files:
        best_days = sorted(successful_files, key=lambda x: x['pnl'], reverse=True)[:5]
        worst_days = sorted(successful_files, key=lambda x: x['pnl'])[:5]
        
        print(f"\n🏆 BEST 5 DAYS")
        for f in best_days:
            print(f"  {f['date']}: P&L {f['pnl']:+,.2f} | {f['trades']} trades | {f['target_hits']} targets")
        
        print(f"\n📉 WORST 5 DAYS")
        for f in worst_days:
            print(f"  {f['date']}: P&L {f['pnl']:+,.2f} | {f['trades']} trades | {f['target_hits']} targets")
    
    print(f"\n{'='*80}\n")

def main():
    """Main function."""
    print("🔧 S3 Status Analyzer and Fix")
    print("="*50)
    
    # Check AWS credentials
    if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
        print("❌ AWS credentials not found!")
        print("Please set:")
        print("  export AWS_ACCESS_KEY_ID='your_key'")
        print("  export AWS_SECRET_ACCESS_KEY='your_secret'")
        print("  export AWS_REGION='us-east-1'")
        return
    
    try:
        # Generate detailed status
        print_detailed_status()
        
        # Generate JSON for API
        status_json = generate_status_json()
        
        print("\n✅ Status analysis complete!")
        print(f"  Generated JSON file with {len(status_json['files'])} file records")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()
