from src.config.database import Base, engine
from src.models import account, audit, confirmation, dataset, review, run, slot_lock  # noqa: F401


def bootstrap():
    Base.metadata.create_all(engine)
