from pydantic import BaseModel, ConfigDict, Field


class ConfirmationCreate(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    shot_id: str = Field(min_length=1, max_length=64)
    version: int = Field(ge=1, le=1_000_000)
    occlusion_low: float = Field(ge=0, le=100)
    occlusion_high: float = Field(ge=0, le=100)
    rig_low: float = Field(ge=0, le=30)
    rig_high: float = Field(ge=0, le=30)


class LockCreate(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    shot_id: str = Field(min_length=1, max_length=64)
    start_minute: int = Field(ge=0, le=1440)
    finish_minute: int = Field(ge=1, le=1440)
