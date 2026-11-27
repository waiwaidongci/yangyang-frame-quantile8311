"""Register all SQLAlchemy tables in every entry point using this package."""

from .account import Account
from .audit import Audit
from .confirmation import Confirmation
from .dataset import Dataset
from .review import Review
from .run import Run
from .slot_lock import SlotLock

__all__ = ['Account', 'Audit', 'Confirmation', 'Dataset', 'Review', 'Run', 'SlotLock']
