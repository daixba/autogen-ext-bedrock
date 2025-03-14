from typing import Awaitable, Callable, Dict, List, Literal, Optional, Union

from autogen_core import ComponentModel
from autogen_core.models import ModelCapabilities, ModelInfo  # type: ignore
from pydantic import BaseModel
from typing_extensions import Required, TypedDict


class CreateArgumentsConfigModel(BaseModel):
    max_tokens: int | None = None
    stop: str | List[str] | None = None
    temperature: float | None = None
    top_p: float | None = None


class BedrockClientConfigurationConfigModel(CreateArgumentsConfigModel):
    model: str
    access_key: str | None = None
    secret_key: str | None = None
    region: str | None = None
    timeout: float | None = None
    max_retries: int | None = None
