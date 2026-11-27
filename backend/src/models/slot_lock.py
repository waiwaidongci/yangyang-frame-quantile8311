from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.config.database import Base


class SlotLock(Base):
    __tablename__ = 'slot_locks'
    __table_args__ = (UniqueConstraint('run_id', 'shot_id'),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey('analysis_runs.id'), index=True)
    shot_id: Mapped[str] = mapped_column(String(64))
    start_minute: Mapped[int] = mapped_column()
    finish_minute: Mapped[int] = mapped_column()
    created_by: Mapped[int] = mapped_column(ForeignKey('accounts.id'))
    created_at: Mapped[str] = mapped_column(String(32))
