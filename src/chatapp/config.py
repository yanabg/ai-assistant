"""Application configuration, loaded once from the environment / ``.env``.

:class:`Config` is the single source of truth for environment-derived
settings. Build it once at startup with :func:`load_config` and pass it down
explicitly; nothing else in the app should touch ``os.environ``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["AIClientConfig", "Config", "load_config"]


class AIClientConfig(BaseModel):
    """The configuration slice the AI client is allowed to use."""

    model_config = ConfigDict(
        frozen=True,
        protected_namespaces=(),
    )

    model_name: str
    api_base_url: str
    api_key: SecretStr
    # request_timeout_s: float
    # max_history_messages: int
    system_prompt: str | None
    data_dir: Path


class Config(BaseSettings):
    """Typed application settings, populated from ``CHATAPP_*`` env vars / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="CHATAPP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_name: str = Field(
        default="gpt-4o-mini",
        description="Identifier of the model the AI client should target.",
    )
    api_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL of the OpenAI-compatible endpoint.",
    )
    api_key: SecretStr = Field(
        description="Secret API token used to authenticate with the endpoint.",
    )

    request_timeout_s: float = Field(
        default=60.0,
        gt=0.0,
        description="Maximum duration of an AI request in seconds.",
    )

    max_history_messages: int = Field(
        default=40,
        ge=2,
        description="Maximum number of previous conversation messages.",
    )

    system_prompt_path: Path | None = Field(
        default=None,
        description="Optional path to a file containing the system prompt.",
    )
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory under which all hierarchies and entries are stored.",
    )

    @field_validator("data_dir")
    @classmethod
    def _absolute_data_dir(cls, value: Path) -> Path:
        """Expand ``~`` and resolve relative paths from the working directory."""
        return value.expanduser().resolve()

    def ensure_data_dir(self) -> Path:
        """Create the configured data directory if it does not exist."""
        path = self.data_dir

        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            message = f"CHATAPP_DATA_DIR is not usable: {path} ({exc})"
            raise RuntimeError(message) from exc

        if not path.is_dir():
            message = f"CHATAPP_DATA_DIR is not a directory: {path}"
            raise RuntimeError(message)

        return path

    def ai_client_config(self) -> AIClientConfig:
        """Project this config down to the slice the AI client receives."""
        return AIClientConfig(
            model_name=self.model_name,
            api_base_url=self.api_base_url,
            api_key=self.api_key,
            # request_timeout_s=self.request_timeout_s,
            # max_history_messages=self.max_history_messages,
            system_prompt=self._read_system_prompt(),
            data_dir=self.data_dir,
        )

    def _read_system_prompt(self) -> str | None:
        """Read the configured system prompt file."""
        path = self.system_prompt_path

        if path is None:
            return None

        if not path.is_file():
            message = f"CHATAPP_SYSTEM_PROMPT_PATH points to a missing file: {path}"
            raise ValueError(message)

        return path.read_text(encoding="utf-8")


def load_config() -> Config:
    """Instantiate :class:`Config` from the process environment and ``.env``.

    Raises:
        pydantic.ValidationError: If a required setting is missing or malformed.
    """
    config = Config()
    config.ensure_data_dir()
    return config
