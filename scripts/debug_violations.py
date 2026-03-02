"""Debug anti-bias violations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.data.cleaner import merge_all_sources
from src.features.builder import build_features
from src.signals.confluence import ConfluenceEngine
from src.backtest.engine import BacktestEngine

config = load_config()
merged = merge_all_sources(config)
features = build_features(merged, config)

confluence = ConfluenceEngine(config)
signals = confluence.calculate_scores(features)

engine = BacktestEngine(config)
result = engine.run(features, signals, "v1")

# Find violations
for i, t in enumerate(result["trades"]):
    if t.exit_bar >= len(features) or t.exit_reason == "end_of_data":
        continue
    bar = features.iloc[t.exit_bar]
    tolerance = (bar["high"] - bar["low"]) * 0.01
    if t.exit_price < bar["low"] - tolerance or t.exit_price > bar["high"] + tolerance:
        print(f"VIOLATION Trade {i}:")
        print(f"  direction: {'LONG' if t.direction == 1 else 'SHORT'}")
        print(f"  exit_reason: {t.exit_reason}")
        print(f"  exit_price: {t.exit_price:.2f}")
        print(f"  bar low: {bar['low']:.2f}, high: {bar['high']:.2f}, close: {bar['close']:.2f}")
        print(f"  entry_price: {t.entry_price:.2f}")
        print(f"  stop_loss: {t.stop_loss:.2f}")
        print(f"  tp1: {t.tp1:.2f}, tp2: {t.tp2:.2f}")
        print(f"  bars_held: {t.bars_held}")
        print()
