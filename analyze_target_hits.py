"""
Analyze Target Hits Summary
===========================
Reads the aggregated CSV and provides the requested summary:
- How many days with 0 target hits
- List of those dates
- Overall target hit statistics
"""

import pandas as pd
import sys
from datetime import datetime

def analyze_target_hits(csv_path):
    """Analyze target hits from aggregated CSV."""
    df = pd.read_csv(csv_path)
    
    # Convert date column if needed
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Daily aggregation
    daily = df.groupby('date').agg(
        trades=('trade_pnl', 'count'),
        target_hits=('exit_type', lambda x: (x == 'TARGET').sum()),
        daily_pnl=('trade_pnl', 'sum'),
        winners=('trade_pnl', lambda x: (x > 0).sum()),
        losers=('trade_pnl', lambda x: (x < 0).sum()),
    ).reset_index()
    
    # Overall stats
    total_days = len(daily)
    days_with_targets = (daily['target_hits'] > 0).sum()
    days_without_targets = (daily['target_hits'] == 0).sum()
    total_trades = daily['trades'].sum()
    total_target_hits = daily['target_hits'].sum()
    total_pnl = daily['daily_pnl'].sum()
    
    # Days with 0 target hits
    zero_target_days = daily[daily['target_hits'] == 0]['date'].tolist()
    
    # Print summary
    print("="*70)
    print("  TARGET HITS ANALYSIS")
    print("="*70)
    print(f"\n📊 OVERALL SUMMARY")
    print(f"  Total Trading Days     : {total_days}")
    print(f"  Days with Target Hits : {days_with_targets}")
    print(f"  Days with 0 Targets   : {days_without_targets}")
    print(f"  Total Trades          : {total_trades}")
    print(f"  Total Target Hits     : {total_target_hits}")
    print(f"  Total P&L             : {total_pnl:+.2f} pts")
    
    print(f"\n🎯 DAYS WITH 0 TARGET HITS ({days_without_targets} days):")
    if zero_target_days:
        for date in sorted(zero_target_days):
            day_data = daily[daily['date'] == date].iloc[0]
            print(f"  {date}: {day_data['trades']} trades, P&L: {day_data['daily_pnl']:+.2f} pts")
    else:
        print("  None! All days had at least one target hit.")
    
    # Target hit distribution
    print(f"\n📈 TARGET HIT DISTRIBUTION:")
    target_dist = daily['target_hits'].value_counts().sort_index()
    for hits, days in target_dist.items():
        print(f"  {hits} target hits: {days} days")
    
    # Days with most target hits
    print(f"\n🏆 TOP 5 DAYS BY TARGET HITS:")
    top_days = daily.nlargest(5, 'target_hits')[['date', 'target_hits', 'trades', 'daily_pnl']]
    for _, row in top_days.iterrows():
        print(f"  {row['date']}: {row['target_hits']} targets, {row['trades']} trades, P&L: {row['daily_pnl']:+.2f}")
    
    # Days with highest P&L
    print(f"\n💰 TOP 5 DAYS BY P&L:")
    top_pnl = daily.nlargest(5, 'daily_pnl')[['date', 'daily_pnl', 'target_hits', 'trades']]
    for _, row in top_pnl.iterrows():
        print(f"  {row['date']}: P&L {row['daily_pnl']:+.2f}, {row['target_hits']} targets, {row['trades']} trades")
    
    print(f"\n{'='*70}\n")
    
    return {
        'total_days': total_days,
        'days_with_targets': days_with_targets,
        'days_without_targets': days_without_targets,
        'zero_target_days': zero_target_days,
        'total_target_hits': total_target_hits,
        'total_pnl': total_pnl,
    }

if __name__ == '__main__':
    csv_path = '/Users/sreenathreddy/Downloads/UniTrader-project/results/s3-aggregated_2024-06-01_to_2024-12-31.csv'
    results = analyze_target_hits(csv_path)
