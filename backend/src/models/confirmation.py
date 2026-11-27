from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.config.database import Base


class Confirmation(Base):
    __tablename__ = 'range_confirmations'
    __table_args__ = (UniqueConstraint('run_id', 'shot_id', 'version'),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey('analysis_runs.id'), index=True)
    shot_id: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column()
    occlusion_low: Mapped[float] = mapped_column()
    occlusion_high: Mapped[float] = mapped_column()
    rig_low: Mapped[float] = mapped_column()
    rig_high: Mapped[float] = mapped_column()
    created_by: Mapped[int] = mapped_column(ForeignKey('accounts.id'))
    created_at: Mapped[str] = mapped_column(String(32))
