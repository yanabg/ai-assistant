"""The seam between the UI and the (to-be-implemented) AI backend.

Only the *contract* is fixed here. The implementation is intentionally left to
the backend author.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI
from pydantic import BaseModel, SecretStr

from chatapp.events import AIEvent, TextDelta
from chatapp.messages import Message

__all__ = ["AIClient"]


class GetHierarchiesArgs(BaseModel):
    pass


class GetEntriesArgs(BaseModel):
    hierarchy: str


@dataclass(frozen=True)
class _Tool:
    """A registered tool with its description, arguments model, and handler."""

    description: str
    args_type: type[BaseModel]
    handler: Callable[..., object] | None

# Build the handler:


TOOLS_REGISTER: dict[str, _Tool] = {
    "get_hierarchies": _Tool(
        description="Use this tool to retrieve all hierarchies.",
        args_type=GetHierarchiesArgs,
        handler=None,
    ),
    "get_entries": _Tool(
        description="Use this tool to retrieve all entries in a given hierarchy.",
        args_type=GetEntriesArgs,
        handler=None,
    ),
}

AI_TOOLS: list[Any] = [
    {
        "type": "function",
        "name": tool_name,
        "description": tool_props.description,
        "parameters": {
            **tool_props.args_type.model_json_schema(),
            "additionalProperties": False,
        },
        "strict": True,
    }
    for tool_name, tool_props in TOOLS_REGISTER.items()
]


class AIClientConfig(Protocol):
    """The narrow slice of configuration the AI client depends on.

    Declared structurally (a ``Protocol``) so the client never reaches for
    globals and can be constructed against a lightweight fake in tests.
    :class:`~chatapp.config.Config` satisfies it automatically.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def api_base_url(self) -> str: ...

    @property
    def api_key(self) -> SecretStr: ...

    @property
    def system_prompt(self) -> str | None: ...

    @property
    def data_dir(self) -> Path: ...


class AIClient:
    """Streaming client for the chat backend.

    .. note::
       This class is intentionally **not implemented** — it is the seam the
       backend author fills in. Only the contract is fixed:

       * construction takes configuration explicitly (no global lookups), and
       * :meth:`send` returns an async iterator of typed :data:`AIEvent` values.
    """

    def __init__(self, config: AIClientConfig) -> None:
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key.get_secret_value(), base_url=config.api_base_url
        )

    async def send(
        self,
        history: Sequence[Message],
        user_message: str,
    ) -> AsyncGenerator[AIEvent]:
        """Stream the assistant's reply to ``user_message``.

        Args:
            history: Prior turns, oldest first, excluding ``user_message``.
            user_message: The turn being answered.

        Yields:
            Events from :data:`~chatapp.events.AIEvent`.
        """
        # Build menaingful history for th emdoel:
        interaction_input: list[Any] = []

        interaction_input.append(
            {
                "role": "system",
                "content": "You are a helpful personal assistant. Reply directly with text.",
            }
        )

        interaction_input.extend({"role": item.role, "content": item.content} for item in history)
        interaction_input.append({"role": "user", "content": user_message})

        # TODO: streaming!
        response = self._client.responses.create(
            model=self._config.model_name,
            input=interaction_input,
            # tool_choice="none"
            tools=AI_TOOLS,
        )

        print(response)
        # yield TextDelta(text = response.output_text)

        # print("OUTPUT TEXT:", repr(response.output_text))
        # print("OUTPUT ITEMS:", response.output)

        yield TextDelta(text=response.output_text)
