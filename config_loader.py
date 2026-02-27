"""
External Configuration Loader
=============================
Load strategy configuration from environment variables or JSON file.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional
from datetime import time

@dataclass
class ExternalConfig:
    """External configuration that can override defaults."""
    base_target: Optional[float] = None
    signal_start_time: Optional[str] = None
    force_exit_time: Optional[str] = None
    allow_opposite_entry: Optional[bool] = None
    daily_target: Optional[float] = None
    historical_candles: Optional[int] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None

def load_config_from_env() -> ExternalConfig:
    """Load config from environment variables."""
    return ExternalConfig(
        base_target=os.getenv('STRATEGY_BASE_TARGET') and float(os.getenv('STRATEGY_BASE_TARGET')),
        signal_start_time=os.getenv('STRATEGY_SIGNAL_START_TIME'),
        force_exit_time=os.getenv('STRATEGY_FORCE_EXIT_TIME'),
        allow_opposite_entry=os.getenv('STRATEGY_ALLOW_OPPOSITE_ENTRY') == 'true',
        daily_target=os.getenv('STRATEGY_DAILY_TARGET') and float(os.getenv('STRATEGY_DAILY_TARGET')),
        historical_candles=os.getenv('STRATEGY_HISTORICAL_CANDLES') and int(os.getenv('STRATEGY_HISTORICAL_CANDLES')),
        symbol=os.getenv('STRATEGY_SYMBOL'),
        timeframe=os.getenv('STRATEGY_TIMEFRAME'),
    )

def load_config_from_json(file_path: str) -> ExternalConfig:
    """Load config from JSON file."""
    if not os.path.exists(file_path):
        return ExternalConfig()
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return ExternalConfig(
        base_target=data.get('base_target'),
        signal_start_time=data.get('signal_start_time'),
        force_exit_time=data.get('force_exit_time'),
        allow_opposite_entry=data.get('allow_opposite_entry'),
        daily_target=data.get('daily_target'),
        historical_candles=data.get('historical_candles'),
        symbol=data.get('symbol'),
        timeframe=data.get('timeframe'),
    )

def apply_external_config(strategy_config, external_config: ExternalConfig):
    """Apply external config to strategy config."""
    if external_config.base_target is not None:
        strategy_config.base_target = external_config.base_target
    if external_config.signal_start_time is not None:
        strategy_config.signal_start_time = external_config.signal_start_time
    if external_config.force_exit_time is not None:
        strategy_config.force_exit_time = external_config.force_exit_time
    if external_config.allow_opposite_entry is not None:
        strategy_config.allow_opposite_entry = external_config.allow_opposite_entry
    if external_config.daily_target is not None:
        strategy_config.daily_target = external_config.daily_target
    if external_config.historical_candles is not None:
        strategy_config.historical_candles = external_config.historical_candles
    if external_config.symbol is not None:
        strategy_config.symbol = external_config.symbol
    if external_config.timeframe is not None:
        strategy_config.timeframe = external_config.timeframe

def create_strategy_config(external_config: ExternalConfig = None) -> 'StrategyConfig':
    """Create strategy config with external overrides."""
    from five_ema_strategy import StrategyConfig
    
    # Default config
    config = StrategyConfig(
        base_target=20.0,
        signal_start_time='09:17',
        force_exit_time='15:25',
        allow_opposite_entry=True,
        daily_target=None,
        historical_candles=500,
    )
    
    # Apply external config
    if external_config:
        apply_external_config(config, external_config)
    
    return config

def print_config_summary(config):
    """Print configuration summary."""
    print(f"\n{'='*60}")
    print(f"  STRATEGY CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Symbol           : {config.symbol}")
    print(f"  Timeframe        : {config.timeframe}")
    print(f"  Base Target      : {config.base_target} pts")
    print(f"  Signal Start     : {config.signal_start_time}")
    print(f"  Force Exit       : {config.force_exit_time}")
    print(f"  Opposite Entry   : {'Enabled' if config.allow_opposite_entry else 'Disabled'}")
    print(f"  Daily Target     : {config.daily_target or 'Disabled'}")
    print(f"  Historical Candles: {config.historical_candles}")
    print(f"{'='*60}\n")
