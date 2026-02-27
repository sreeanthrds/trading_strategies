"""
================================================================================
5 EMA AGGRESSIVE INTRADAY STRATEGY - Backtesting Engine v2.0
================================================================================
Updated: 2025-02-26 | Enhanced with DataFrame Analytics & Historical Warm-up

STRATEGY OVERVIEW:
    The 5 EMA Strategy is an aggressive intraday breakout system that uses the
    5-period Exponential Moving Average (EMA) on 1-minute candles to identify
    high-probability entry signals. It incorporates loss recovery through
    dynamic target adjustment and strict position management rules.

SIGNAL DETECTION:
    1. Calculate the 5-period EMA on 1-minute OHLCV data.
    2. Starting from 09:21, scan for "signal candles":
       - A signal candle is one whose entire body (high to low) does NOT touch
         the 5 EMA line (i.e., candle_low > EMA  OR  candle_high < EMA).
    3. Entry trigger:
       - SELL signal: Signal candle closes ABOVE EMA + next candle breaks BELOW
         the signal candle's low.
       - BUY signal:  Signal candle closes BELOW EMA + next candle breaks ABOVE
         the signal candle's high.

ENTRY RULES:
    - SELL entry price = signal candle low (breakout below).
    - BUY  entry price = signal candle high (breakout above).
    - Precise entry is resolved using tick-level data within the entry candle.

EXIT RULES:
    - Target: entry_price +/- current_target points (direction-dependent).
    - Stop Loss: opposite extreme of the signal candle.
      * SELL SL = signal candle high.
      * BUY  SL = signal candle low.
    - Force exit: all open positions are force-closed at FORCE_EXIT_TIME (15:25).
    - Tick-level precision is used to find exact exit timestamps.

LOSS ACCUMULATION (Aggressive Recovery):
    - Base target per trade = 20 points (configurable).
    - On a LOSS: cumulative_loss += abs(loss). Next target = base_target +
      cumulative_loss. This ensures the next winning trade recovers all
      accumulated losses plus the base profit.
    - On a WIN:  cumulative_loss = max(0, cumulative_loss - profit).
      Next target = max(base_target, base_target + remaining cumulative_loss).
    - After all losses are recovered, target resets to the base 20 points.

POSITION MANAGEMENT:
    - Only ONE position per side (BUY or SELL) at a time.
    - allow_opposite_entry flag:
      * True  -> while a BUY is open, a new SELL signal can be taken (and
                 vice versa), effectively allowing hedged positions.
      * False -> no new entries until the current position is closed.
    - After a position exits (target, SL, or force), the same side is
      available for re-entry on the next valid signal.

DAILY TARGET (Optional):
    - daily_target parameter (default: None = disabled).
    - When enabled, if cumulative P&L for the day >= daily_target, all
      positions are force-closed and no further trades are taken.

INDICATOR COMPUTATION (Historical Warm-Up):
    - Before strategy execution, fetch N historical candles (default 500)
      from previous trading days to seed indicator calculations.
    - Compute 5 EMA on the full dataset (historical + current day) upfront.
    - Store EMA as a column in the DataFrame so signal detection uses
      chart-accurate values from the very first candle of the day.
    - Tick data is only fetched AFTER a signal condition is matched, to
      resolve exact entry/exit timestamps within the candle.

DETAILED RESULTS OUTPUT (DataFrame Analytics):
    - All trade results are stored in a pandas DataFrame with 41 rich columns:
      * Signal Analysis: signal_time, signal_type, signal_candle OHLC, range, ema_at_signal
      * Entry Details: entry_time, entry_price, target_price, stop_loss_price, risk_points
      * Exit Details: exit_time, exit_price, exit_type, trade_pnl, cumulative_pnl
      * Trade Analytics: hold_duration, mfe_points/price/time, mae_points/price/time
      * Performance Metrics: r_multiple, target_reached_pct, reward_risk_ratio, is_winner
      * State Tracking: cumulative_loss_before/after, next_target, ticks_in_trade
    - Tick-level analytics compute Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE)
      with exact timestamps, R-multiple, and percentage of target reached.
    - Enables arbitrary post-hoc analysis, insight generation, and statistical studies.
    - Results auto-exported to CSV/JSON/pickle in the results/ directory.

PARAMETERS (all configurable in StrategyConfig):
    - base_target:          20 points (profit target per trade)
    - force_exit_time:      "15:25" (HH:MM)
    - signal_start_time:    "09:21" (HH:MM)
    - allow_opposite_entry: True
    - daily_target:         None (disabled) or float
    - symbol:               "NIFTY"
    - timeframe:            "1m"
    - historical_candles:   500 (warm-up candles for indicator accuracy)

DATA SOURCES:
    - OHLCV: ClickHouse table `nse_ohlcv_indices` (1-minute bars).
    - Ticks:  ClickHouse table `nse_ticks_indices` (tick-by-tick LTP data).
    - Connection via HTTP API at 34.200.220.45:8123, database `tradelayout`.

PERFORMANCE METRICS (automatically computed):
    - Win Rate, Avg Win/Loss, Profit Factor
    - Max Drawdown, Peak P&L, Cumulative Returns
    - Avg MFE/MAE, Avg R-Multiple, Avg Target Reach %
    - Hold Time Statistics, Tick Count Analysis
    - Loss Recovery Efficiency, Force Exit Impact

USAGE EXAMPLES:
    # Basic backtest
    config = StrategyConfig(base_target=20.0, historical_candles=500)
    engine = BacktestEngine(config)
    trades_df = engine.run(date(2024, 10, 3))
    
    # With daily target and opposite entries disabled
    config = StrategyConfig(
        base_target=25.0,
        daily_target=100.0,
        allow_opposite_entry=False,
        historical_candles=1000
    )
    
    # Export results
    ReportGenerator.export_csv(trades_df, "my_results.csv")
    ReportGenerator.export_json(trades_df, "results.json")

FILE STRUCTURE (single file, modular sections):
    1. Configuration          - StrategyConfig dataclass with all parameters
    2. Data Layer             - ClickHouseDataFetcher (OHLCV + historical + tick queries)
    3. Signal Detection       - SignalDetector (EMA calc + signal candle scan)
    4. Tick Precision Engine  - TickPrecisionEngine (entry/exit + MFE/MAE analytics)
    5. Trade / Position Mgmt  - PositionManager (open/close, side tracking, loss recovery)
    6. Backtest Engine        - BacktestEngine (orchestrates everything, returns DataFrame)
    7. Reporting              - ReportGenerator (detailed prints + export capabilities)
    8. Main                   - CLI entry point with auto-export
================================================================================
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import requests
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class StrategyConfig:
    """All tunable parameters for the 5 EMA strategy."""

    # --- Connection ---
    clickhouse_host: str = '34.200.220.45'
    clickhouse_port: str = '8123'
    clickhouse_user: str = 'default'
    clickhouse_password: str = ''
    clickhouse_database: str = 'tradelayout'

    # --- Symbol & Timeframe ---
    symbol: str = 'NIFTY'
    timeframe: str = '1m'

    # --- Strategy Parameters ---
    base_target: float = 20.0            # Final daily target (cumulative P&L goal)
    max_target_cap: float = 30.0         # Maximum target cap per side (individual trade target never exceeds this)
    signal_start_time: str = '09:17'     # HH:MM - start scanning for signals
    force_exit_time: str = '15:25'       # HH:MM - force close all positions
    tick_data_end_time: str = '16:07'    # HH:MM - tick data availability cutoff

    # --- Position Management ---
    allow_opposite_entry: bool = True    # Allow entry on opposite side while a position is open
    max_positions_per_side: int = 1      # Max simultaneous positions per side (BUY/SELL)

    # --- Daily Target (optional) ---
    daily_target: Optional[float] = None  # If set, stop trading when cumulative P&L >= this

    # --- Indicator Warm-Up ---
    historical_candles: int = 500        # Number of historical candles before current day for EMA warm-up

    # --- Backtest Scan Limits ---
    max_exit_scan_minutes: int = 390     # Max minutes to scan forward for exit (full day)

    def get_signal_start_dt(self, base_date) -> datetime:
        """Return signal start as a datetime for the given date."""
        h, m = map(int, self.signal_start_time.split(':'))
        return base_date.replace(hour=h, minute=m, second=0, microsecond=0)

    def get_force_exit_dt(self, base_date) -> datetime:
        """Return force exit as a datetime for the given date."""
        h, m = map(int, self.force_exit_time.split(':'))
        return base_date.replace(hour=h, minute=m, second=0, microsecond=0)

    def get_tick_end_dt(self, base_date) -> datetime:
        """Return tick data end as a datetime for the given date."""
        h, m = map(int, self.tick_data_end_time.split(':'))
        return base_date.replace(hour=h, minute=m, second=0, microsecond=0)


# ============================================================================
# 2. DATA LAYER - ClickHouse HTTP Fetcher
# ============================================================================

class ClickHouseDataFetcher:
    """Handles all data retrieval from ClickHouse via HTTP API."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self._base_url = f"http://{config.clickhouse_host}:{config.clickhouse_port}"
        self._params = {
            'database': config.clickhouse_database,
            'user': config.clickhouse_user,
            'password': config.clickhouse_password,
        }

    # ------------------------------------------------------------------
    # Connection Test
    # ------------------------------------------------------------------
    def test_connection(self) -> bool:
        """Verify ClickHouse is reachable."""
        try:
            resp = requests.get(self._base_url, timeout=10)
            if resp.status_code == 200:
                print("Connected to ClickHouse HTTP interface.")
                return True
            print(f"ClickHouse returned status {resp.status_code}")
            return False
        except Exception as e:
            print(f"ClickHouse connection failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Generic query helper
    # ------------------------------------------------------------------
    def _execute_query(self, query: str, timeout: int = 30) -> Optional[pd.DataFrame]:
        """Execute a query and return a DataFrame (or None on failure)."""
        try:
            params = {**self._params, 'query': query}
            resp = requests.post(self._base_url, params=params, timeout=timeout)
            resp.raise_for_status()

            lines = resp.text.strip().split('\n')
            if len(lines) < 2:
                return None

            columns = lines[0].split('\t')
            rows = []
            for line in lines[1:]:
                if line.strip():
                    vals = line.split('\t')
                    if len(vals) == len(columns):
                        rows.append(vals)
            if not rows:
                return None
            return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            print(f"Query error: {e}")
            return None

    # ------------------------------------------------------------------
    # OHLCV Data
    # ------------------------------------------------------------------
    def fetch_ohlcv(self, date, symbol: str = None, timeframe: str = None) -> pd.DataFrame:
        """Fetch 1-minute OHLCV bars for a given date."""
        symbol = symbol or self.config.symbol
        timeframe = timeframe or self.config.timeframe

        query = (
            f"SELECT timestamp, open, high, low, close, volume "
            f"FROM nse_ohlcv_indices "
            f"WHERE symbol = '{symbol}' "
            f"AND timeframe = '{timeframe}' "
            f"AND toDate(timestamp) = '{date}' "
            f"ORDER BY timestamp ASC "
            f"FORMAT TabSeparatedWithNames"
        )
        df = self._execute_query(query)
        if df is None:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        df['volume'] = df['volume'].astype(int)
        print(f"Fetched {len(df)} {timeframe} candles for {date}")
        return df

    # ------------------------------------------------------------------
    # Historical OHLCV (for indicator warm-up)
    # ------------------------------------------------------------------
    def fetch_historical_candles(self, before_date, num_candles: int = 500,
                                  symbol: str = None, timeframe: str = None) -> pd.DataFrame:
        """Fetch N historical candles immediately before a given date.

        This retrieves candles from previous trading days to seed indicator
        calculations (e.g., EMA) so that the first candle of the current day
        has an accurate indicator value.
        """
        symbol = symbol or self.config.symbol
        timeframe = timeframe or self.config.timeframe

        query = (
            f"SELECT timestamp, open, high, low, close, volume "
            f"FROM nse_ohlcv_indices "
            f"WHERE symbol = '{symbol}' "
            f"AND timeframe = '{timeframe}' "
            f"AND toDate(timestamp) < '{before_date}' "
            f"ORDER BY timestamp DESC "
            f"LIMIT {num_candles} "
            f"FORMAT TabSeparatedWithNames"
        )
        df = self._execute_query(query)
        if df is None:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        df['volume'] = df['volume'].astype(int)

        # Reverse to chronological order (query was DESC)
        df = df.iloc[::-1].reset_index(drop=True)
        print(f"Fetched {len(df)} historical candles for indicator warm-up "
              f"(from {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()})")
        return df

    # ------------------------------------------------------------------
    # Tick Data - Full Day (single query, load once)
    # ------------------------------------------------------------------
    def fetch_full_day_ticks(self, date, symbol: str = None) -> pd.DataFrame:
        """Fetch ALL tick data for an entire trading day in a single query.

        This is the optimized replacement for per-minute tick fetching.
        Returns a DataFrame with columns: timestamp, ltp, ltq
        sorted by timestamp ascending, filtered to 09:15:00 - 15:30:00.
        Fills missing seconds with forward fill and keeps only last tick per second.
        """
        symbol = symbol or self.config.symbol
        query = (
            f"SELECT timestamp, ltp, ltq "
            f"FROM nse_ticks_indices "
            f"WHERE symbol = '{symbol}' "
            f"AND toDate(timestamp) = '{date}' "
            f"AND timestamp >= toDateTime('{date} 09:15:00') "
            f"AND timestamp <= toDateTime('{date} 15:30:00') "
            f"ORDER BY timestamp ASC "
            f"FORMAT TabSeparatedWithNames"
        )
        df = self._execute_query(query, timeout=120)
        if df is None or len(df) == 0:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['ltp'] = df['ltp'].astype(float)
        df['ltq'] = df['ltq'].astype(int)
        
        # Keep only the last tick of each second, but sum the quantities
        aggregated = df.groupby(df['timestamp'].dt.floor('s')).agg({
            'timestamp': 'last',
            'ltp': 'last',
            'ltq': 'sum'
        }).reset_index(drop=True)
        
        # Create complete time range from 09:15:00 to 15:30:00
        start_time = pd.to_datetime(f'{date} 09:15:00')
        end_time = pd.to_datetime(f'{date} 15:30:00')
        full_time_range = pd.date_range(start=start_time, end=end_time, freq='s')
        
        # Create complete DataFrame with all seconds
        full_df = pd.DataFrame({'timestamp': full_time_range})
        
        # Merge with actual ticks
        merged = full_df.merge(aggregated, on='timestamp', how='left')
        
        # Forward fill missing prices but set quantity to 0 for missing seconds
        merged['ltp'] = merged['ltp'].ffill()
        merged['ltq'] = merged['ltq'].fillna(0)
        
        # Drop any rows that still have NaN (shouldn't happen if we have at least one tick)
        merged = merged.dropna()
        
        print(f"Fetched {len(df)} raw ticks for {date} "
              f"({df['timestamp'].iloc[0].strftime('%H:%M:%S')} to "
              f"{df['timestamp'].iloc[-1].strftime('%H:%M:%S')})")
        print(f"After processing: {len(merged)} ticks (1 per second, ltq summed, ffill applied)")
        
        return merged

    # ------------------------------------------------------------------
    # Tick Data (single minute) - kept for backward compatibility
    # ------------------------------------------------------------------
    def fetch_ticks_for_minute(self, minute_ts: datetime, symbol: str = None) -> pd.DataFrame:
        """Fetch tick data for a specific 1-minute window."""
        symbol = symbol or self.config.symbol
        end_ts = minute_ts + timedelta(minutes=1)
        query = (
            f"SELECT timestamp, ltp, ltq "
            f"FROM nse_ticks_indices "
            f"WHERE symbol = '{symbol}' "
            f"AND timestamp >= '{minute_ts}' AND timestamp < '{end_ts}' "
            f"ORDER BY timestamp ASC "
            f"FORMAT TabSeparatedWithNames"
        )
        df = self._execute_query(query)
        if df is None:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['ltp'] = df['ltp'].astype(float)
        df['ltq'] = df['ltq'].astype(int)
        return df

    # ------------------------------------------------------------------
    # Tick Data (time range)
    # ------------------------------------------------------------------
    def fetch_ticks_range(self, start: datetime, end: datetime, symbol: str = None) -> pd.DataFrame:
        """Fetch tick data for an arbitrary time range."""
        symbol = symbol or self.config.symbol
        query = (
            f"SELECT timestamp, ltp, ltq "
            f"FROM nse_ticks_indices "
            f"WHERE symbol = '{symbol}' "
            f"AND timestamp > '{start}' AND timestamp < '{end}' "
            f"ORDER BY timestamp ASC "
            f"FORMAT TabSeparatedWithNames"
        )
        df = self._execute_query(query)
        if df is None:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['ltp'] = df['ltp'].astype(float)
        df['ltq'] = df['ltq'].astype(int)
        return df

    # ------------------------------------------------------------------
    # Single-candle OHLCV lookup (for exit scanning)
    # ------------------------------------------------------------------
    def fetch_single_candle(self, candle_ts: datetime, symbol: str = None) -> Optional[Dict]:
        """Fetch high/low for a single 1-minute candle."""
        symbol = symbol or self.config.symbol
        query = (
            f"SELECT high, low FROM nse_ohlcv_indices "
            f"WHERE symbol = '{symbol}' AND timeframe = '1m' "
            f"AND timestamp = '{candle_ts}' "
            f"FORMAT TabSeparatedWithNames"
        )
        df = self._execute_query(query)
        if df is None:
            return None
        return {'high': float(df.iloc[0]['high']), 'low': float(df.iloc[0]['low'])}


# ============================================================================
# 3. SIGNAL DETECTION
# ============================================================================

class SignalDetector:
    """Calculates 5 EMA and identifies signal candles + entry triggers."""

    @staticmethod
    def calculate_5ema(df: pd.DataFrame) -> pd.DataFrame:
        """Add 5-period EMA column to OHLCV DataFrame."""
        df = df.copy()
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        return df

    @staticmethod
    def identify_signals(df: pd.DataFrame, start_time: datetime,
                         end_time: datetime = None) -> List[Dict]:
        """
        Scan OHLCV data for 5 EMA signal candles.

        Returns a list of signal dicts with type, prices, and candle references.
        Signals after end_time (force exit time) are excluded.
        """
        signals = []

        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i - 1]

            # Enforce time window (only consider signals from start_time onwards)
            if current['timestamp'] < start_time:
                continue
            if end_time and current['timestamp'] >= end_time:
                break

            prev_high = prev['high']
            prev_low = prev['low']
            prev_ema = prev['ema5']

            # Signal candle must NOT touch the 5 EMA
            candle_above_ema = prev_low > prev_ema
            candle_below_ema = prev_high < prev_ema

            if not (candle_above_ema or candle_below_ema):
                continue

            signal = None

            # SELL: signal candle above EMA, next candle breaks below signal low
            if candle_above_ema and prev_low > prev_ema and current['low'] < prev_low:
                signal = {
                    'type': 'SELL',
                    'signal_candle_idx': i - 1,
                    'entry_candle_idx': i,
                    'signal_candle': prev,
                    'entry_candle': current,
                    'signal_high': prev_high,
                    'signal_low': prev_low,
                    'signal_close': prev['close'],
                    'entry_price': prev_low,
                    'stop_loss': prev_high,
                }

            # BUY: signal candle below EMA, next candle breaks above signal high
            elif candle_below_ema and prev_high < prev_ema and current['high'] > prev_high:
                signal = {
                    'type': 'BUY',
                    'signal_candle_idx': i - 1,
                    'entry_candle_idx': i,
                    'signal_candle': prev,
                    'entry_candle': current,
                    'signal_high': prev_high,
                    'signal_low': prev_low,
                    'signal_close': prev['close'],
                    'entry_price': prev_high,
                    'stop_loss': prev_low,
                }

            if signal:
                print(f"  {signal['type']} signal @ {prev['timestamp']}, "
                      f"entry candle @ {current['timestamp']}")
                signals.append(signal)

        print(f"Total signals detected: {len(signals)}")
        return signals


# ============================================================================
# 4. TICK PRECISION ENGINE
# ============================================================================

class TickPrecisionEngine:
    """Resolves exact entry and exit timestamps using pre-loaded tick data.

    OPTIMIZED: All tick data for the day is loaded once via load_day_ticks().
    All lookups use numpy array operations on the pre-loaded DataFrame,
    eliminating per-minute API calls (~150+ calls → 1 call per day).
    """

    def __init__(self, data_fetcher: ClickHouseDataFetcher, config: StrategyConfig):
        self.fetcher = data_fetcher
        self.config = config
        self._day_ticks: pd.DataFrame = pd.DataFrame()
        self._tick_timestamps: np.ndarray = np.array([])
        self._tick_prices: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # Load full day tick data (call once per day)
    # ------------------------------------------------------------------
    def load_day_ticks(self, date, symbol: str = None):
        """Pre-load all tick data for the day. Must be called before other methods."""
        self._day_ticks = self.fetcher.fetch_full_day_ticks(date, symbol)
        if not self._day_ticks.empty:
            self._tick_timestamps = self._day_ticks['timestamp'].values
            self._tick_prices = self._day_ticks['ltp'].values
        else:
            self._tick_timestamps = np.array([])
            self._tick_prices = np.array([])

    def _get_ticks_between(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Slice pre-loaded ticks between start and end timestamps."""
        if self._day_ticks.empty:
            return pd.DataFrame()
        start_np = np.datetime64(start)
        end_np = np.datetime64(end)
        mask = (self._tick_timestamps > start_np) & (self._tick_timestamps < end_np)
        return self._day_ticks[mask]

    def _get_ticks_from(self, start: datetime) -> pd.DataFrame:
        """Slice pre-loaded ticks from start timestamp onwards."""
        if self._day_ticks.empty:
            return pd.DataFrame()
        start_np = np.datetime64(start)
        mask = self._tick_timestamps >= start_np
        return self._day_ticks[mask]

    def _get_ticks_in_minute(self, minute_ts: datetime) -> pd.DataFrame:
        """Slice pre-loaded ticks for a specific 1-minute window."""
        if self._day_ticks.empty:
            return pd.DataFrame()
        start_np = np.datetime64(minute_ts)
        end_np = np.datetime64(minute_ts + timedelta(minutes=1))
        mask = (self._tick_timestamps >= start_np) & (self._tick_timestamps < end_np)
        return self._day_ticks[mask]

    # ------------------------------------------------------------------
    # Find precise entry tick
    # ------------------------------------------------------------------
    def find_entry_tick(self, signal: Dict) -> Optional[Dict]:
        """
        Find the exact tick where entry_price is breached.

        For SELL: first tick where ltp <= entry_price (signal_low).
        For BUY:  first tick where ltp >= entry_price (signal_high).
        """
        search_start = signal['entry_candle']['timestamp'].replace(second=0, microsecond=0)
        search_end = search_start + timedelta(minutes=1)

        entry_price = signal['entry_price']
        is_sell = signal['type'] == 'SELL'

        # Get ticks in the search window
        start_np = np.datetime64(search_start)
        end_np = np.datetime64(search_end)
        mask = (self._tick_timestamps >= start_np) & (self._tick_timestamps < end_np)
        window_prices = self._tick_prices[mask]
        window_indices = np.where(mask)[0]

        if len(window_prices) == 0:
            print(f"  No ticks for entry search from {search_start}")
            return None

        # Find first tick matching entry condition
        if is_sell:
            hits = window_prices <= entry_price
        else:
            hits = window_prices >= entry_price

        hit_indices = np.where(hits)[0]
        if len(hit_indices) == 0:
            print(f"  Entry price {entry_price} not hit in ticks from {search_start}")
            return None

        # Get the first matching tick
        idx = window_indices[hit_indices[0]]
        tick = self._day_ticks.iloc[idx]
        return {
            'timestamp': tick['timestamp'],
            'price': tick['ltp'],
            'ltq': tick['ltq'],
        }

    # ------------------------------------------------------------------
    # Find precise exit tick (target / SL / force exit)
    # ------------------------------------------------------------------
    def find_exit_tick(self, entry_tick: Dict, signal_type: str,
                       target_price: float, stop_loss: float,
                       force_exit_dt: datetime) -> Optional[Dict]:
        """
        Scan forward from entry tick through pre-loaded ticks to find:
          1. Target price hit
          2. Stop loss price hit
          3. Force exit time reached

        Directly scans tick data (no OHLCV candle pass needed).
        """
        entry_time = entry_tick['timestamp']
        is_sell = signal_type == 'SELL'

        # Get all ticks after entry
        entry_np = np.datetime64(entry_time)
        force_np = np.datetime64(force_exit_dt)
        mask = self._tick_timestamps > entry_np
        post_entry_indices = np.where(mask)[0]

        if len(post_entry_indices) == 0:
            print(f"  No ticks after entry time {entry_time}")
            return None

        # Scan through post-entry ticks
        for idx in post_entry_indices:
            ts = self._tick_timestamps[idx]
            price = self._tick_prices[idx]

            # Force exit check
            if ts >= force_np:
                return self._resolve_force_exit(force_exit_dt)

            # Check target
            if is_sell and price <= target_price:
                tick = self._day_ticks.iloc[idx]
                return {'timestamp': tick['timestamp'], 'price': price, 'exit_type': 'TARGET'}
            if not is_sell and price >= target_price:
                tick = self._day_ticks.iloc[idx]
                return {'timestamp': tick['timestamp'], 'price': price, 'exit_type': 'TARGET'}

            # Check stop loss
            if is_sell and price >= stop_loss:
                tick = self._day_ticks.iloc[idx]
                return {'timestamp': tick['timestamp'], 'price': price, 'exit_type': 'STOP_LOSS'}
            if not is_sell and price <= stop_loss:
                tick = self._day_ticks.iloc[idx]
                return {'timestamp': tick['timestamp'], 'price': price, 'exit_type': 'STOP_LOSS'}

        print(f"  No exit found within tick data")
        return None

    # ------------------------------------------------------------------
    # Force exit resolution
    # ------------------------------------------------------------------
    def _resolve_force_exit(self, force_exit_dt: datetime) -> Optional[Dict]:
        """Get the LTP at or after force exit time from pre-loaded ticks."""
        force_np = np.datetime64(force_exit_dt)
        mask = self._tick_timestamps >= force_np
        indices = np.where(mask)[0]

        if len(indices) > 0:
            idx = indices[0]
            tick = self._day_ticks.iloc[idx]
            return {
                'timestamp': tick['timestamp'],
                'price': tick['ltp'],
                'exit_type': 'FORCE_EXIT',
            }

        # Fallback: use last available tick before force exit
        if not self._day_ticks.empty:
            last_tick = self._day_ticks.iloc[-1]
            return {
                'timestamp': last_tick['timestamp'],
                'price': last_tick['ltp'],
                'exit_type': 'FORCE_EXIT',
            }

        print(f"  No ticks at force exit time {force_exit_dt}")
        return None

    # ------------------------------------------------------------------
    # Post-trade analytics (MFE / MAE / R-multiple etc.)
    # ------------------------------------------------------------------
    def compute_trade_analytics(self, entry_tick: Dict, exit_tick: Dict,
                                 signal_type: str, target_price: float,
                                 stop_loss: float) -> Dict:
        """
        Compute trade analytics from pre-loaded tick data:
          - Max Favorable Excursion (MFE): best unrealized P&L during trade
          - Max Adverse Excursion (MAE): worst unrealized P&L during trade
          - MFE/MAE timestamps and prices
          - R-multiple: actual P&L / risk (entry-to-SL distance)
          - Target reach %: how far towards target price went
          - Tick counts
        """
        entry_time = entry_tick['timestamp']
        exit_time = exit_tick['timestamp']
        entry_price = entry_tick['price']
        is_sell = signal_type == 'SELL'

        analytics = {
            'mfe_points': 0.0,
            'mae_points': 0.0,
            'mfe_price': entry_price,
            'mae_price': entry_price,
            'mfe_time': entry_time,
            'mae_time': entry_time,
            'ticks_in_trade': 0,
            'time_to_mfe_seconds': 0.0,
            'time_to_mae_seconds': 0.0,
            'r_multiple': 0.0,
            'target_reached_pct': 0.0,
        }

        # Slice ticks between entry and exit from pre-loaded data
        entry_np = np.datetime64(entry_time)
        exit_np = np.datetime64(exit_time)
        mask = (self._tick_timestamps > entry_np) & (self._tick_timestamps < exit_np)
        trade_indices = np.where(mask)[0]

        if len(trade_indices) == 0:
            return analytics

        trade_prices = self._tick_prices[trade_indices]
        trade_timestamps = self._tick_timestamps[trade_indices]
        analytics['ticks_in_trade'] = len(trade_prices)

        # Vectorized unrealized P&L computation
        if is_sell:
            unrealized = entry_price - trade_prices
        else:
            unrealized = trade_prices - entry_price

        # MFE: max favorable
        mfe_idx = np.argmax(unrealized)
        analytics['mfe_points'] = float(unrealized[mfe_idx])
        analytics['mfe_price'] = float(trade_prices[mfe_idx])
        mfe_ts = pd.Timestamp(trade_timestamps[mfe_idx])
        analytics['mfe_time'] = mfe_ts
        analytics['time_to_mfe_seconds'] = (mfe_ts - entry_time).total_seconds()

        # MAE: max adverse (min of unrealized)
        mae_idx = np.argmin(unrealized)
        analytics['mae_points'] = float(unrealized[mae_idx])
        analytics['mae_price'] = float(trade_prices[mae_idx])
        mae_ts = pd.Timestamp(trade_timestamps[mae_idx])
        analytics['mae_time'] = mae_ts
        analytics['time_to_mae_seconds'] = (mae_ts - entry_time).total_seconds()

        # R-multiple: pnl / risk where risk = |entry - SL|
        risk = abs(entry_price - stop_loss)
        actual_pnl = (entry_price - exit_tick['price']) if is_sell else (
            exit_tick['price'] - entry_price)
        if risk > 0:
            analytics['r_multiple'] = round(actual_pnl / risk, 3)

        # Target reach %: how far towards target the price went
        target_distance = abs(target_price - entry_price)
        if target_distance > 0:
            analytics['target_reached_pct'] = round(
                (analytics['mfe_points'] / target_distance) * 100, 2)

        return analytics


# ============================================================================
# 5. POSITION MANAGER
# ============================================================================

class PositionManager:
    """
    Tracks open positions and enforces:
      - Max one position per side.
      - allow_opposite_entry flag.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        # Track open positions: {'BUY': <trade_dict or None>, 'SELL': <trade_dict or None>}
        self.open_positions: Dict[str, Optional[Dict]] = {'BUY': None, 'SELL': None}

    def can_enter(self, side: str) -> bool:
        """Check if a new entry on the given side is allowed."""
        # Same side already occupied?
        if self.open_positions[side] is not None:
            return False

        # Opposite side occupied and opposite entries not allowed?
        opposite = 'SELL' if side == 'BUY' else 'BUY'
        if self.open_positions[opposite] is not None and not self.config.allow_opposite_entry:
            return False

        return True

    def open_position(self, side: str, trade_info: Dict):
        """Register an open position."""
        self.open_positions[side] = trade_info

    def close_position(self, side: str):
        """Close a position on the given side."""
        self.open_positions[side] = None

    def get_open_sides(self) -> List[str]:
        """Return list of sides that currently have open positions."""
        return [s for s, pos in self.open_positions.items() if pos is not None]

    def has_any_open(self) -> bool:
        return any(pos is not None for pos in self.open_positions.values())


# ============================================================================
# 6. BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """
    Orchestrates the full single-day backtest:
      1. Fetch historical + current day OHLCV data.
      2. Compute EMA upfront on the combined dataset.
      3. Filter to current day for signal detection.
      4. For each signal, check position rules, compute target, find entry/exit.
      5. Compute detailed trade analytics (MFE, MAE, R-multiple, etc.).
      6. Track cumulative loss and adjust targets.
      7. Enforce force exit and optional daily target.
      8. Return results as a detailed pandas DataFrame.
    """

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.fetcher = ClickHouseDataFetcher(self.config)
        self.tick_engine = TickPrecisionEngine(self.fetcher, self.config)

    # ------------------------------------------------------------------
    # Prepare data: historical warm-up + current day + upfront EMA
    # ------------------------------------------------------------------
    def _prepare_data(self, date, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch historical candles + current day, calculate EMA on combined,
        return (full_df_with_ema, current_day_df_with_ema).
        """
        # Fetch current day
        current_df = self.fetcher.fetch_ohlcv(date, symbol)
        if current_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Fetch historical candles for warm-up
        hist_df = self.fetcher.fetch_historical_candles(
            date, self.config.historical_candles, symbol
        )

        # Combine: historical first, then current day
        if not hist_df.empty:
            combined_df = pd.concat([hist_df, current_df], ignore_index=True)
            hist_count = len(hist_df)
        else:
            combined_df = current_df.copy()
            hist_count = 0
            print("WARNING: No historical candles fetched. EMA may be inaccurate for early candles.")

        # Compute EMA upfront on the full combined dataset
        combined_df = SignalDetector.calculate_5ema(combined_df)

        # Split back: current day only (with accurate EMA values)
        current_day_df = combined_df.iloc[hist_count:].reset_index(drop=True)

        print(f"EMA computed on {len(combined_df)} candles "
              f"({hist_count} historical + {len(current_day_df)} current day)")

        return combined_df, current_day_df

    # ------------------------------------------------------------------
    # Build detailed trade dict with all analytics
    # ------------------------------------------------------------------
    def _build_trade_record(self, trade_no: int, side: str, signal: Dict,
                            entry_tick: Dict, exit_tick: Dict,
                            target_price: float, stop_loss: float,
                            target_points_used: float, cumulative_loss_before: float,
                            cumulative_loss_after: float, cumulative_pnl: float,
                            pnl: float, next_target: float,
                            analytics: Dict) -> Dict:
        """Construct a comprehensive trade record dict."""
        hold_duration = exit_tick['timestamp'] - entry_tick['timestamp']
        risk_points = abs(entry_tick['price'] - stop_loss)
        signal_candle_range = signal['signal_high'] - signal['signal_low']

        return {
            # --- Identifiers ---
            'trade_no': trade_no,
            'date': entry_tick['timestamp'].date(),

            # --- Signal Info ---
            'signal_time': signal['signal_candle']['timestamp'],
            'signal_type': side,
            'signal_candle_open': signal['signal_candle']['open'],
            'signal_candle_high': signal['signal_high'],
            'signal_candle_low': signal['signal_low'],
            'signal_candle_close': signal['signal_close'],
            'signal_candle_range': signal_candle_range,
            'ema_at_signal': signal['signal_candle']['ema5'],

            # --- Entry Info ---
            'entry_candle_time': signal['entry_candle']['timestamp'],
            'entry_time': entry_tick['timestamp'],
            'entry_price': entry_tick['price'],

            # --- Target / SL ---
            'target_price': target_price,
            'stop_loss_price': stop_loss,
            'target_points_used': target_points_used,
            'risk_points': risk_points,

            # --- Exit Info ---
            'exit_time': exit_tick['timestamp'],
            'exit_price': exit_tick['price'],
            'exit_type': exit_tick['exit_type'],
            'exit_side': 'BUY' if side == 'SELL' else 'SELL',

            # --- P&L ---
            'trade_pnl': round(pnl, 2),
            'cumulative_pnl': round(cumulative_pnl, 2),
            'hold_duration': hold_duration,
            'hold_duration_seconds': hold_duration.total_seconds(),

            # --- Loss Accumulation State ---
            'cumulative_loss_before': round(cumulative_loss_before, 2),
            'cumulative_loss_after': round(cumulative_loss_after, 2),
            'next_target': round(next_target, 2),

            # --- MFE / MAE Analytics ---
            'mfe_points': round(analytics['mfe_points'], 2),
            'mae_points': round(analytics['mae_points'], 2),
            'mfe_price': analytics['mfe_price'],
            'mae_price': analytics['mae_price'],
            'mfe_time': analytics['mfe_time'],
            'mae_time': analytics['mae_time'],
            'time_to_mfe_seconds': analytics['time_to_mfe_seconds'],
            'time_to_mae_seconds': analytics['time_to_mae_seconds'],
            'ticks_in_trade': analytics['ticks_in_trade'],

            # --- Derived Analytics ---
            'r_multiple': analytics['r_multiple'],
            'target_reached_pct': analytics['target_reached_pct'],
            'reward_risk_ratio': round(target_points_used / risk_points, 3) if risk_points > 0 else 0.0,
            'is_winner': pnl > 0,
        }

    # ------------------------------------------------------------------
    # Main backtest runner
    # ------------------------------------------------------------------
    def run(self, date, symbol: str = None) -> pd.DataFrame:
        """Run backtest for a single day. Returns detailed trades DataFrame."""
        symbol = symbol or self.config.symbol
        print(f"\n{'='*80}")
        print(f"BACKTESTING: {date}  |  Symbol: {symbol}  |  "
              f"Final Target: {self.config.base_target} pts  |  "
              f"Max Cap: {self.config.max_target_cap} pts  |  "
              f"Historical: {self.config.historical_candles} candles")
        print(f"{'='*80}")

        # --- Fetch & prepare data with historical warm-up ---
        full_df, current_day_df = self._prepare_data(date, symbol)
        if current_day_df.empty:
            print(f"No data for {date}")
            return pd.DataFrame()

        base_dt = current_day_df['timestamp'].iloc[0]
        start_dt = self.config.get_signal_start_dt(base_dt)
        force_exit_dt = self.config.get_force_exit_dt(base_dt)

        # --- Load full day tick data (single query, major performance gain) ---
        self.tick_engine.load_day_ticks(date, symbol)

        # =================================================================
        # CANDLE-BY-CANDLE PROCESSING (V2 - Shared Target Pool with Cap)
        #
        # target_points is a GLOBAL shared variable, updated on every exit.
        # At exit time we use the CURRENT target_points (not the value at entry).
        #
        # V2 CHANGES:
        #   - target_points is capped at max_target_cap (e.g. 30 pts)
        #   - When cumulative_pnl >= base_target (e.g. 20 pts), exit day
        #   - Losses dynamically added to target pool
        #   - On exit, remaining balance shared; other side adjusts target
        # =================================================================
        target_points = min(self.config.base_target, self.config.max_target_cap)
        cumulative_loss = 0.0
        cumulative_pnl = 0.0
        open_positions: Dict[str, Dict] = {}      # side -> position info
        trades: List[Dict] = []
        trade_counter = 0
        daily_target_hit = False

        for i in range(1, len(current_day_df)):
            if daily_target_hit:
                break

            current = current_day_df.iloc[i]
            prev = current_day_df.iloc[i - 1]
            candle_time = current['timestamp']

            if candle_time < start_dt:
                continue
            if candle_time >= force_exit_dt:
                break

            # ----------------------------------------------------------
            # A) Candle-level quick check: might any exit happen here?
            # ----------------------------------------------------------
            exits_possible = False
            for side, pos in open_positions.items():
                sig_ep = pos['signal_entry_price']  # signal level, not tick price
                sl = pos['stop_loss']
                # Use base_target (minimum possible) for conservative check
                # so we never miss an exit if target_points decreased mid-candle
                min_tp = self.config.base_target
                if side == 'BUY':
                    tp_check = sig_ep + min_tp
                    if current['high'] >= tp_check or current['low'] <= sl:
                        exits_possible = True
                else:
                    tp_check = sig_ep - min_tp
                    if current['low'] <= tp_check or current['high'] >= sl:
                        exits_possible = True

            # ----------------------------------------------------------
            # B) Signal detection on this candle
            # ----------------------------------------------------------
            signal = None
            prev_ema = prev['ema5']
            prev_high = prev['high']
            prev_low = prev['low']
            candle_above_ema = prev_low > prev_ema
            candle_below_ema = prev_high < prev_ema

            if candle_above_ema and current['low'] < prev_low:
                signal = {
                    'type': 'SELL',
                    'signal_candle_idx': i - 1,
                    'entry_candle_idx': i,
                    'signal_candle': prev,
                    'entry_candle': current,
                    'signal_high': prev_high,
                    'signal_low': prev_low,
                    'signal_close': prev['close'],
                    'entry_price': prev_low,
                    'stop_loss': prev_high,
                }
            elif candle_below_ema and current['high'] > prev_high:
                signal = {
                    'type': 'BUY',
                    'signal_candle_idx': i - 1,
                    'entry_candle_idx': i,
                    'signal_candle': prev,
                    'entry_candle': current,
                    'signal_high': prev_high,
                    'signal_low': prev_low,
                    'signal_close': prev['close'],
                    'entry_price': prev_high,
                    'stop_loss': prev_low,
                }

            # Can we enter?
            entry_possible = False
            if signal:
                sig_side = signal['type']
                blocked = False
                if sig_side in open_positions:
                    blocked = True
                else:
                    opposite = 'SELL' if sig_side == 'BUY' else 'BUY'
                    if opposite in open_positions and not self.config.allow_opposite_entry:
                        blocked = True
                if blocked:
                    print(f"  Skipping {sig_side} signal @ {candle_time} "
                          f"- position rules block entry")
                else:
                    entry_possible = True

            # Nothing to do in this candle → skip tick processing
            if not exits_possible and not entry_possible:
                continue

            # ----------------------------------------------------------
            # C) Tick-by-tick processing for this candle
            # ----------------------------------------------------------
            ticks = self.tick_engine._get_ticks_in_minute(candle_time)
            if ticks.empty:
                continue

            entry_done = False
            for _, tick in ticks.iterrows():
                if daily_target_hit:
                    break

                tick_price = tick['ltp']
                tick_time = tick['timestamp']

                # --- 1. Check exits first (using CURRENT target_points) ---
                for side in list(open_positions.keys()):
                    pos = open_positions[side]
                    sig_ep = pos['signal_entry_price']  # signal level for target calc
                    tick_ep = pos['entry_tick']['price']  # actual fill for P&L
                    sl = pos['stop_loss']
                    tp = sig_ep + target_points if side == 'BUY' else sig_ep - target_points

                    hit_target = False
                    hit_sl = False
                    if side == 'BUY':
                        hit_target = tick_price >= tp
                        hit_sl = tick_price <= sl
                    else:
                        hit_target = tick_price <= tp
                        hit_sl = tick_price >= sl

                    if not (hit_target or hit_sl):
                        continue

                    # --- EXIT ---
                    exit_type = 'TARGET' if hit_target else 'STOP_LOSS'
                    exit_tick = {
                        'timestamp': tick_time,
                        'price': tick_price,
                        'ltq': tick['ltq'],
                        'exit_type': exit_type,
                    }

                    if side == 'BUY':
                        pnl = tick_price - tick_ep
                    else:
                        pnl = tick_ep - tick_price

                    loss_before = cumulative_loss
                    tp_used = target_points  # record BEFORE P&L update
                    if pnl < 0:
                        cumulative_loss += abs(pnl)
                    else:
                        cumulative_loss = max(0.0, cumulative_loss - pnl)
                    cumulative_pnl += pnl

                    # V2: Recalculate target = remaining to reach final target
                    # remaining = base_target - cumulative_pnl (what we still need)
                    # But also add any accumulated losses that haven't been recovered
                    remaining_to_goal = self.config.base_target - cumulative_pnl
                    raw_target = max(self.config.base_target, remaining_to_goal)
                    target_points = min(raw_target, self.config.max_target_cap)

                    analytics = self.tick_engine.compute_trade_analytics(
                        pos['entry_tick'], exit_tick, side, tp, sl
                    )
                    trade = self._build_trade_record(
                        trade_no=pos['trade_no'],
                        side=side, signal=pos['signal'],
                        entry_tick=pos['entry_tick'], exit_tick=exit_tick,
                        target_price=tp, stop_loss=sl,
                        target_points_used=tp_used,
                        cumulative_loss_before=loss_before,
                        cumulative_loss_after=cumulative_loss,
                        cumulative_pnl=cumulative_pnl,
                        pnl=pnl, next_target=target_points,
                        analytics=analytics,
                    )
                    trades.append(trade)
                    del open_positions[side]

                    print(f"  >> {exit_type} {side} @ {tick_time} | "
                          f"P&L: {pnl:+.2f} | Cum: {cumulative_pnl:+.2f} | "
                          f"Next Target: {target_points:.1f} (cap: {self.config.max_target_cap})")

                    # V2: Check final daily target (base_target is the daily goal)
                    if cumulative_pnl >= self.config.base_target:
                        print(f"\n  FINAL TARGET of {self.config.base_target} pts "
                              f"reached! Cum P&L: {cumulative_pnl:.2f}. Stopping.")
                        daily_target_hit = True

                    # Also check optional daily_target if set
                    elif (self.config.daily_target is not None
                            and cumulative_pnl >= self.config.daily_target):
                        print(f"\n  DAILY TARGET of {self.config.daily_target} pts "
                              f"reached! Cum P&L: {cumulative_pnl:.2f}. Stopping.")
                        daily_target_hit = True

                # --- 2. Check entry (after all exits on this tick) ---
                if entry_possible and not entry_done and not daily_target_hit:
                    sig_side = signal['type']
                    entry_price = signal['entry_price']

                    # Re-check can_enter (exits above may have freed a slot)
                    can_enter = True
                    if sig_side in open_positions:
                        can_enter = False
                    else:
                        opposite = 'SELL' if sig_side == 'BUY' else 'BUY'
                        if opposite in open_positions and not self.config.allow_opposite_entry:
                            can_enter = False

                    if can_enter:
                        entered = False
                        if sig_side == 'BUY' and tick_price >= entry_price:
                            entered = True
                        elif sig_side == 'SELL' and tick_price <= entry_price:
                            entered = True

                        if entered:
                            trade_counter += 1
                            entry_tick = {
                                'timestamp': tick_time,
                                'price': tick_price,
                                'ltq': tick['ltq'],
                            }
                            open_positions[sig_side] = {
                                'entry_tick': entry_tick,
                                'signal_entry_price': entry_price,  # signal level for target
                                'stop_loss': signal['stop_loss'],
                                'signal': signal,
                                'trade_no': trade_counter,
                            }
                            tp = (entry_price + target_points if sig_side == 'BUY'
                                  else entry_price - target_points)
                            print(f"\n  #{trade_counter} {sig_side} Entry @ {tick_time}"
                                  f" | Price: {tick_price:.2f}"
                                  f" | Target: {tp:.2f} ({target_points:.1f} pts, cap: {self.config.max_target_cap})"
                                  f" | SL: {signal['stop_loss']:.2f}"
                                  f" | Need: {self.config.base_target - cumulative_pnl:.1f} to final")
                            entry_done = True

        # --- Force exit remaining open positions ---
        for side in list(open_positions.keys()):
            pos = open_positions[side]
            print(f"\n  Force-closing open {side} position at {force_exit_dt}")
            exit_tick = self.tick_engine._resolve_force_exit(force_exit_dt)
            if exit_tick:
                sig_ep = pos['signal_entry_price']
                tick_ep = pos['entry_tick']['price']
                sl = pos['stop_loss']
                tp = sig_ep + target_points if side == 'BUY' else sig_ep - target_points

                if side == 'BUY':
                    pnl = exit_tick['price'] - tick_ep
                else:
                    pnl = tick_ep - exit_tick['price']

                loss_before = cumulative_loss
                tp_used = target_points  # record BEFORE P&L update
                if pnl < 0:
                    cumulative_loss += abs(pnl)
                else:
                    cumulative_loss = max(0.0, cumulative_loss - pnl)
                cumulative_pnl += pnl

                # V2: Recalculate target with cap
                remaining_to_goal = self.config.base_target - cumulative_pnl
                raw_target = max(self.config.base_target, remaining_to_goal)
                target_points = min(raw_target, self.config.max_target_cap)

                analytics = self.tick_engine.compute_trade_analytics(
                    pos['entry_tick'], exit_tick, side, tp, sl
                )
                trade = self._build_trade_record(
                    trade_no=pos['trade_no'],
                    side=side, signal=pos['signal'],
                    entry_tick=pos['entry_tick'], exit_tick=exit_tick,
                    target_price=tp, stop_loss=sl,
                    target_points_used=tp_used,
                    cumulative_loss_before=loss_before,
                    cumulative_loss_after=cumulative_loss,
                    cumulative_pnl=cumulative_pnl,
                    pnl=pnl, next_target=target_points,
                    analytics=analytics,
                )
                trades.append(trade)
            del open_positions[side]

        # Sort trades by entry time for chronological order
        trades.sort(key=lambda t: t['entry_time'])

        # Convert to DataFrame
        if trades:
            trades_df = pd.DataFrame(trades)
            return trades_df
        return pd.DataFrame()

    def run_multi_day(self, dates: List, symbol: str = None) -> pd.DataFrame:
        """Run backtest over multiple dates. Returns combined trades DataFrame."""
        all_trades = []
        for d in dates:
            day_df = self.run(d, symbol)
            if not day_df.empty:
                all_trades.append(day_df)
        if all_trades:
            return pd.concat(all_trades, ignore_index=True)
        return pd.DataFrame()


# ============================================================================
# 7. REPORTING
# ============================================================================

class ReportGenerator:
    """DataFrame-based trade reporting with export capabilities."""

    @staticmethod
    def print_trades(trades_df: pd.DataFrame, title: str = "TRADE RESULTS"):
        """Pretty-print each trade from the results DataFrame."""
        if trades_df.empty:
            print("\nNo trades executed.\n")
            return

        print(f"\n{'='*120}")
        print(f"  {title}")
        print(f"{'='*120}")

        for _, t in trades_df.iterrows():
            print(f"\n  TRADE #{t['trade_no']}  [{t['signal_type']}]")
            print(f"  {'─'*60}")
            print(f"    Signal : {t['signal_time']}  | EMA: {t['ema_at_signal']:.2f}  "
                  f"| Range: {t['signal_candle_range']:.2f}")
            print(f"    Entry  : {t['entry_time']}  @ {t['entry_price']:.2f}")
            print(f"    Target : {t['target_price']:.2f}  |  "
                  f"Stop Loss: {t['stop_loss_price']:.2f}  |  "
                  f"Risk: {t['risk_points']:.2f} pts")
            print(f"    Exit   : {t['exit_time']}  @ {t['exit_price']:.2f}  "
                  f"[{t['exit_type']}]")
            print(f"    Hold   : {t['hold_duration']}  "
                  f"({t['hold_duration_seconds']:.0f}s)")
            pnl_sym = "+" if t['trade_pnl'] >= 0 else ""
            print(f"    P&L    : {pnl_sym}{t['trade_pnl']:.2f} pts  |  "
                  f"Cum P&L: {t['cumulative_pnl']:.2f} pts  |  "
                  f"R: {t['r_multiple']:.2f}")
            print(f"    MFE    : +{t['mfe_points']:.2f} pts @ {t['mfe_time']}  |  "
                  f"MAE: {t['mae_points']:.2f} pts @ {t['mae_time']}")
            print(f"    Target Reached: {t['target_reached_pct']:.1f}%  |  "
                  f"Ticks: {t['ticks_in_trade']}  |  "
                  f"Next Target: {t['next_target']:.1f} pts")

        ReportGenerator._print_summary(trades_df)

    @staticmethod
    def _print_summary(trades_df: pd.DataFrame):
        """Print summary statistics from the trades DataFrame."""
        print(f"\n{'─'*120}")
        print(f"  SUMMARY")
        print(f"{'─'*120}")

        total = len(trades_df)
        winners = trades_df[trades_df['trade_pnl'] > 0]
        losers = trades_df[trades_df['trade_pnl'] < 0]
        force_exits = trades_df[trades_df['exit_type'] == 'FORCE_EXIT']
        target_hits = trades_df[trades_df['exit_type'] == 'TARGET']
        final_pnl = trades_df['cumulative_pnl'].iloc[-1]

        print(f"    Total Trades     : {total}")
        if total:
            print(f"    Winners          : {len(winners)}  "
                  f"({len(winners)/total*100:.1f}%)")
        print(f"    Losers           : {len(losers)}")
        print(f"    Target Hits      : {len(target_hits)}")
        print(f"    Force Exits      : {len(force_exits)}")
        print(f"    Total Profit     : {winners['trade_pnl'].sum():+.2f} pts")
        print(f"    Total Loss       : {losers['trade_pnl'].sum():+.2f} pts")
        print(f"    Final P&L        : {final_pnl:+.2f} pts")

        if len(winners):
            print(f"    Avg Win          : {winners['trade_pnl'].mean():.2f} pts")
        if len(losers):
            print(f"    Avg Loss         : {losers['trade_pnl'].mean():.2f} pts")
        if total:
            print(f"    Avg Hold Time    : "
                  f"{trades_df['hold_duration_seconds'].mean()/60:.1f} min")
            print(f"    Avg MFE          : "
                  f"+{trades_df['mfe_points'].mean():.2f} pts")
            print(f"    Avg MAE          : "
                  f"{trades_df['mae_points'].mean():.2f} pts")
            print(f"    Avg R-Multiple   : "
                  f"{trades_df['r_multiple'].mean():.3f}")
            print(f"    Avg Target Reach : "
                  f"{trades_df['target_reached_pct'].mean():.1f}%")
            print(f"    Max Drawdown     : "
                  f"{trades_df['cumulative_pnl'].min():.2f} pts")
            print(f"    Peak P&L         : "
                  f"{trades_df['cumulative_pnl'].max():.2f} pts")

        # --- Daily breakdown: target hits, P&L per day ---
        if 'date' in trades_df.columns:
            daily = trades_df.groupby('date').agg(
                trades=('trade_pnl', 'count'),
                target_hits=('exit_type', lambda x: (x == 'TARGET').sum()),
                daily_pnl=('trade_pnl', 'sum'),
            ).reset_index()

            days_with_targets = (daily['target_hits'] > 0).sum()
            days_without_targets = (daily['target_hits'] == 0).sum()
            total_days = len(daily)

            print(f"\n{'─'*120}")
            print(f"  DAILY BREAKDOWN")
            print(f"{'─'*120}")
            print(f"    Total Trading Days       : {total_days}")
            print(f"    Days with Target Hits     : {days_with_targets}")
            print(f"    Days with 0 Target Hits   : {days_without_targets}")
            print(f"\n    {'Date':<14} {'Trades':>7} {'Targets':>8} {'P&L':>10}")
            print(f"    {'─'*42}")
            for _, row in daily.iterrows():
                pnl_str = f"{row['daily_pnl']:+.2f}"
                print(f"    {row['date']:<14} {row['trades']:>7} "
                      f"{row['target_hits']:>8} {pnl_str:>10}")

        print(f"\n{'='*120}\n")

    @staticmethod
    def print_dataframe_info(trades_df: pd.DataFrame):
        """Print DataFrame column list and shape for reference."""
        if trades_df.empty:
            return
        print(f"\n  DataFrame Shape: {trades_df.shape[0]} rows x {trades_df.shape[1]} columns")
        print(f"  Columns: {list(trades_df.columns)}\n")

    @staticmethod
    def export_csv(trades_df: pd.DataFrame, filepath: str):
        """Export trades DataFrame to CSV."""
        if trades_df.empty:
            print("No trades to export.")
            return
        trades_df.to_csv(filepath, index=False)
        print(f"  Exported {len(trades_df)} trades to {filepath}")

    @staticmethod
    def export_json(trades_df: pd.DataFrame, filepath: str):
        """Export trades DataFrame to JSON."""
        if trades_df.empty:
            print("No trades to export.")
            return
        trades_df.to_json(filepath, orient='records', indent=2, date_format='iso')
        print(f"  Exported {len(trades_df)} trades to {filepath}")

    @staticmethod
    def export_pickle(trades_df: pd.DataFrame, filepath: str):
        """Export trades DataFrame to pickle for fast Python reuse."""
        if trades_df.empty:
            print("No trades to export.")
            return
        trades_df.to_pickle(filepath)
        print(f"  Exported {len(trades_df)} trades to {filepath}")


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    # --- Configuration (modify as needed) ---
    config = StrategyConfig(
        base_target=20.0,
        signal_start_time='09:17',
        force_exit_time='15:25',
        allow_opposite_entry=True,
        daily_target=None,          # Set to e.g. 50.0 to enable daily target
        historical_candles=500,     # Warm-up candles for accurate EMA
    )

    # --- Test connection ---
    fetcher = ClickHouseDataFetcher(config)
    if not fetcher.test_connection():
        print("Cannot connect to ClickHouse. Exiting.")
        return

    # --- Run backtest from June 1 to Dec 31 ---
    engine = BacktestEngine(config)
    
    # Date range: June 1 to December 31, 2024
    start_date = datetime(2024, 6, 1).date()
    end_date = datetime(2024, 12, 31).date()
    
    print(f"\nStarting 5 EMA Strategy Backtest")
    print(f"Date Range       : {start_date} to {end_date}")
    print(f"Symbol           : {config.symbol}")
    print(f"Timeframe        : {config.timeframe}")
    print(f"Base Target      : {config.base_target} pts")
    print(f"Force Exit       : {config.force_exit_time}")
    print(f"Opposite Entry   : {'Enabled' if config.allow_opposite_entry else 'Disabled'}")
    print(f"Daily Target     : {config.daily_target or 'Disabled'}")
    print(f"Historical Candles: {config.historical_candles}")
    print(f"{'-'*50}")

    # Collect all trades across the date range
    all_trades = []
    current_date = start_date
    total_days = 0
    successful_days = 0
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() < 5:  # Monday=0, Friday=4
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
    
    # Combine all daily results
    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
        print(f"\n{'='*50}")
        print(f"BACKTEST COMPLETE")
        print(f"{'='*50}")
        print(f"Total Trading Days: {total_days}")
        print(f"Days with Trades   : {successful_days}")
        print(f"Total Trades       : {len(trades_df)}")
    else:
        trades_df = pd.DataFrame()
        print(f"\nNo trades executed in the entire period.")

    # --- Print detailed report ---
    if not trades_df.empty:
        ReportGenerator.print_trades(trades_df)
        ReportGenerator.print_dataframe_info(trades_df)

        # --- Export results ---
        out_dir = '/Users/sreenathreddy/Downloads/UniTrader-project/results'
        os.makedirs(out_dir, exist_ok=True)
        csv_path = f"{out_dir}/trades_{start_date}_to_{end_date}.csv"
        ReportGenerator.export_csv(trades_df, csv_path)
        
        # Also export summary statistics
        summary_path = f"{out_dir}/summary_{start_date}_to_{end_date}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"5 EMA Strategy Backtest Summary\n")
            f.write(f"{'='*60}\n")
            f.write(f"Period: {start_date} to {end_date}\n")
            f.write(f"Total Trading Days: {total_days}\n")
            f.write(f"Days with Trades: {successful_days}\n")
            f.write(f"Total Trades: {len(trades_df)}\n")
            if not trades_df.empty:
                final_pnl = trades_df['cumulative_pnl'].iloc[-1]
                winners = trades_df[trades_df['trade_pnl'] > 0]
                losers = trades_df[trades_df['trade_pnl'] < 0]
                target_hits = trades_df[trades_df['exit_type'] == 'TARGET']
                f.write(f"Final P&L: {final_pnl:+.2f} pts\n")
                f.write(f"Winners: {len(winners)} | Losers: {len(losers)}\n")
                f.write(f"Total Target Hits: {len(target_hits)}\n")
                f.write(f"Total Profit: {winners['trade_pnl'].sum():+.2f} pts\n")
                f.write(f"Total Loss: {losers['trade_pnl'].sum():+.2f} pts\n\n")

                # Daily breakdown
                daily = trades_df.groupby('date').agg(
                    trades=('trade_pnl', 'count'),
                    target_hits=('exit_type', lambda x: (x == 'TARGET').sum()),
                    daily_pnl=('trade_pnl', 'sum'),
                ).reset_index()

                days_with_targets = int((daily['target_hits'] > 0).sum())
                days_without_targets = int((daily['target_hits'] == 0).sum())

                f.write(f"{'='*60}\n")
                f.write(f"DAILY BREAKDOWN\n")
                f.write(f"{'='*60}\n")
                f.write(f"Days with Target Hits   : {days_with_targets}\n")
                f.write(f"Days with 0 Target Hits : {days_without_targets}\n\n")
                f.write(f"{'Date':<14} {'Trades':>7} {'Targets':>8} {'P&L':>10}\n")
                f.write(f"{'-'*42}\n")
                for _, row in daily.iterrows():
                    f.write(f"{row['date']:<14} {row['trades']:>7} "
                            f"{row['target_hits']:>8} {row['daily_pnl']:>+10.2f}\n")

                # Also export daily breakdown as CSV
                daily_csv_path = f"{out_dir}/daily_summary_{start_date}_to_{end_date}.csv"
                daily.to_csv(daily_csv_path, index=False)
                print(f"  Daily summary saved to {daily_csv_path}")

        print(f"  Summary saved to {summary_path}")

    return trades_df


if __name__ == "__main__":
    main()
