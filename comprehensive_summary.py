"""
Comprehensive Daily Summary
===========================
Detailed summary of all trading days with performance metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

def comprehensive_daily_summary(csv_path):
    """Generate comprehensive summary for all trading days."""
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
    
    # Sort by date
    daily = daily.sort_values('date')
    
    # Calculate additional metrics
    daily['win_rate'] = (daily['winners'] / daily['trades'] * 100).round(2)
    daily['target_hit_rate'] = (daily['target_hits'] / daily['trades'] * 100).round(2)
    daily['profit_factor'] = (daily['total_profit'] / abs(daily['total_loss'])).round(2).replace([np.inf, -np.inf], 0)
    daily['avg_trade'] = (daily['daily_pnl'] / daily['trades']).round(2)
    daily['risk_reward'] = (abs(daily['avg_win'] / daily['avg_loss'])).round(2).replace([np.inf, -np.inf], 0)
    
    # Monthly aggregation
    daily['month'] = pd.to_datetime(daily['date']).dt.to_period('M')
    monthly = daily.groupby('month').agg(
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
    total_days = len(daily)
    profitable_days = (daily['daily_pnl'] > 0).sum()
    losing_days = (daily['daily_pnl'] < 0).sum()
    flat_days = (daily['daily_pnl'] == 0).sum()
    
    # Print comprehensive summary
    print("="*80)
    print("  COMPREHENSIVE 5EMA STRATEGY SUMMARY")
    print("="*80)
    
    print(f"\n📊 OVERALL PERFORMANCE (Jun 1 - Dec 31, 2024)")
    print(f"  Total Trading Days     : {total_days}")
    print(f"  Profitable Days        : {profitable_days} ({profitable_days/total_days*100:.1f}%)")
    print(f"  Losing Days            : {losing_days} ({losing_days/total_days*100:.1f}%)")
    print(f"  Flat Days              : {flat_days} ({flat_days/total_days*100:.1f}%)")
    print(f"  Total Trades           : {daily['trades'].sum()}")
    print(f"  Total Target Hits      : {daily['target_hits'].sum()}")
    print(f"  Total P&L              : {daily['daily_pnl'].sum():+,.2f} pts")
    print(f"  Total Profit           : {daily['total_profit'].sum():+,.2f} pts")
    print(f"  Total Loss             : {daily['total_loss'].sum():+,.2f} pts")
    print(f"  Overall Win Rate       : {(daily['winners'].sum()/daily['trades'].sum()*100):.1f}%")
    print(f"  Target Hit Rate        : {(daily['target_hits'].sum()/daily['trades'].sum()*100):.1f}%")
    print(f"  Profit Factor          : {(daily['total_profit'].sum()/abs(daily['total_loss'].sum())):.2f}")
    print(f"  Average Daily P&L      : {daily['daily_pnl'].mean():+.2f} pts")
    print(f"  Best Day               : {daily.loc[daily['daily_pnl'].idxmax(), 'daily_pnl']:+,.2f} pts")
    print(f"  Worst Day              : {daily.loc[daily['daily_pnl'].idxmin(), 'daily_pnl']:+,.2f} pts")
    
    print(f"\n📈 MONTHLY BREAKDOWN")
    print(f"{'Month':<10} {'Days':>5} {'Trades':>7} {'Targets':>8} {'P&L':>12} {'Win%':>6} {'Avg/P&L':>9}")
    print(f"{'-'*65}")
    for _, row in monthly.iterrows():
        month_str = str(row['month'])
        pnl_str = f"{row['monthly_pnl']:+,.0f}"
        avg_str = f"{row['avg_daily_pnl']:+,.0f}"
        print(f"{month_str:<10} {row['days']:>5} {row['trades']:>7} {row['target_hits']:>8} {pnl_str:>12} {row['win_rate']:>6.1f}% {avg_str:>9}")
    
    print(f"\n🎯 TARGET HIT ANALYSIS")
    target_dist = daily['target_hits'].value_counts().sort_index()
    print(f"  Target Hits Range     : {daily['target_hits'].min()} - {daily['target_hits'].max()}")
    print(f"  Average Target Hits   : {daily['target_hits'].mean():.1f} per day")
    print(f"  Days with 0 Targets   : {(daily['target_hits'] == 0).sum()} ({(daily['target_hits'] == 0).sum()/total_days*100:.1f}%)")
    print(f"  Days with 10+ Targets : {(daily['target_hits'] >= 10).sum()} ({(daily['target_hits'] >= 10).sum()/total_days*100:.1f}%)")
    
    print(f"\n📋 TOP 10 BEST DAYS")
    best_days = daily.nlargest(10, 'daily_pnl')[['date', 'daily_pnl', 'trades', 'target_hits', 'win_rate']]
    for _, row in best_days.iterrows():
        print(f"  {row['date']}: P&L {row['daily_pnl']:+,.2f} | {row['trades']} trades | {row['target_hits']} targets | {row['win_rate']:.1f}% win")
    
    print(f"\n📉 TOP 10 WORST DAYS")
    worst_days = daily.nsmallest(10, 'daily_pnl')[['date', 'daily_pnl', 'trades', 'target_hits', 'win_rate']]
    for _, row in worst_days.iterrows():
        print(f"  {row['date']}: P&L {row['daily_pnl']:+,.2f} | {row['trades']} trades | {row['target_hits']} targets | {row['win_rate']:.1f}% win")
    
    print(f"\n🏆 PERFORMANCE MILESTONES")
    print(f"  Highest Single Day P&L: {daily['daily_pnl'].max():+,.2f} pts on {daily.loc[daily['daily_pnl'].idxmax(), 'date']}")
    print(f"  Most Trades in a Day  : {daily['trades'].max()} trades on {daily.loc[daily['trades'].idxmax(), 'date']}")
    print(f"  Most Target Hits      : {daily['target_hits'].max()} targets on {daily.loc[daily['target_hits'].idxmax(), 'date']}")
    print(f"  Highest Win Rate       : {daily['win_rate'].max():.1f}% on {daily.loc[daily['win_rate'].idxmax(), 'date']}")
    print(f"  Best Profit Factor    : {daily['profit_factor'].max():.2f} on {daily.loc[daily['profit_factor'].idxmax(), 'date']}")
    
    # Streak analysis
    daily['is_profitable'] = daily['daily_pnl'] > 0
    daily['streak'] = (daily['is_profitable'] != daily['is_profitable'].shift()).cumsum()
    streaks = daily.groupby(['streak', 'is_profitable']).size().unstack(fill_value=0)
    
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
    print(f"  Max Daily Drawdown    : {daily['max_drawdown'].min():+,.2f} pts")
    print(f"  Average Daily Range   : {daily['avg_mfe'].mean() + abs(daily['avg_mae'].mean()):.1f} pts")
    print(f"  Average Risk/Reward   : {daily['risk_reward'].mean():.2f}")
    print(f"  Average R-Multiple    : {daily['avg_r_multiple'].mean():.3f}")
    
    # Save detailed CSV
    output_path = csv_path.replace('.csv', '_detailed_summary.csv')
    daily.to_csv(output_path, index=False)
    print(f"\n💾 Detailed daily summary saved to: {output_path}")
    
    print(f"\n{'='*80}\n")
    
    return daily, monthly

if __name__ == '__main__':
    csv_path = '/Users/sreenathreddy/Downloads/UniTrader-project/results/s3-aggregated_2024-06-01_to_2024-12-31.csv'
    daily, monthly = comprehensive_daily_summary(csv_path)
