"""The seam between the UI and the (to-be-implemented) AI backend.

Only the *contract* is fixed here. The implementation is intentionally left to
the backend author.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI
from pydantic import BaseModel, SecretStr

from chatapp.events import AIEvent, TextDelta, ToolCall, ToolResult
from chatapp.messages import Message

__all__ = ["AIClient"]


class GetHierarchiesArgs(BaseModel):
    pass


class GetEntriesArgs(BaseModel):
    hierarchy: str


# Build the handlers:
def get_hierarchies(raw_args: str, config: AIClientConfig) -> str:
    """List every hierarchy (a directory under ``data_dir``) with its schema.

    Returns a JSON array of ``{"name", "description"}``, where the description
    is the text of the hierarchy's ``SCHEMA.md`` (empty if absent).
    """
    GetHierarchiesArgs.model_validate_json(raw_args or "{}")

    hierarchies: list[dict[str, str]] = []
    data_dir = config.data_dir
    if data_dir.is_dir():
        for child in sorted(data_dir.iterdir()):
            if not child.is_dir():
                continue
            schema = child / "SCHEMA.md"
            description = schema.read_text(encoding="utf-8").strip() if schema.is_file() else ""
            hierarchies.append({"name": child.name, "description": description})

    return json.dumps(hierarchies)


def get_entries(raw_args: str, config: AIClientConfig) -> str:
    """List the entries of a single hierarchy by name.

    Returns a JSON array of ``{"name"}`` — one per ``*.md`` entry file in
    ``data_dir/<hierarchy>``, excluding the ``SCHEMA.md`` schema file.
    """
    args = GetEntriesArgs.model_validate_json(raw_args or "{}")

    entries: list[dict[str, str]] = []
    hierarchy_dir = config.data_dir / args.hierarchy
    if hierarchy_dir.is_dir():
        for child in sorted(hierarchy_dir.iterdir()):
            if child.is_file() and child.suffix == ".md" and child.name != "SCHEMA.md":
                entries.append({"name": child.stem})

    return json.dumps(entries)


def _extract_tool_calls(response: Any) -> list[Any]:
    return [x for x in response.output if x.type == "function_call"]

@dataclass(frozen=True)
class _Tool:
    """A registered tool with its description, arguments model, and handler."""

    description: str
    args_type: type[BaseModel]
    handler: Callable[[str, AIClientConfig], str]


TOOLS_REGISTER: dict[str, _Tool] = {
    "get_hierarchies": _Tool(
        description="Use this tool to retrieve all hierarchies.",
        args_type=GetHierarchiesArgs,
        handler=get_hierarchies,
    ),
    "get_entries": _Tool(
        description="Use this tool to retrieve all entries in a given hierarchy.",
        args_type=GetEntriesArgs,
        handler=get_entries,
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
        # Build meaningful history for the model.
        context: list[Any] = []

        context.append(
            {
                "role": "system",
                "content": self._config.system_prompt,
            }
        )
        context.extend({"role": item.role, "content": item.content} for item in history)
        context.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        response = self._client.responses.create(
            model=self._config.model_name,
            input=context,
            tools=AI_TOOLS,
        )
        tool_calls = _extract_tool_calls(response)

        print(response)

        while tool_calls:
            print(f"Discovered {len(tool_calls)} tool calls that should be executed.")
            for tc in tool_calls:
                print(tc)
            print()

            context.extend(response.output)

            for tc in tool_calls:
                yield ToolCall(
                    name=tc.name,
                    arguments=json.loads(tc.arguments),
                )

                handler = TOOLS_REGISTER[tc.name].handler
                current_result = handler(tc.arguments, self._config)

                yield ToolResult(
                    name=tc.name,
                    content=current_result,
                    is_error=False,
                )

                context.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc.call_id,
                        "output": current_result,
                    }
                )

            response = self._client.responses.create(
                model=self._config.model_name,
                input=context,
                tools=AI_TOOLS,
            )
            tool_calls = _extract_tool_calls(response)

            print(response)

        yield TextDelta(text=response.output_text)
