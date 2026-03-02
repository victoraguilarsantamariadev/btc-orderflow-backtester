"""Base signal interface. All signals must implement this."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseSignal(ABC):
    """Abstract base class for all trading signals.

    Each signal returns a Series with values:
        +weight: Long signal
        -weight: Short signal
        0: No signal
    """

    def __init__(self, config: dict):
        self.config = config
        self.weight = config.get("weight", 1)

    @abstractmethod
    def calculate(self, features: pd.DataFrame) -> pd.Series:
        """Calculate signal from feature DataFrame.

        Must return a Series with the same index as features.
        Values: +weight (long), -weight (short), 0 (no signal).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Signal name for logging and debugging."""
        pass
