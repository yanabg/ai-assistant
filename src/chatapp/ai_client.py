"""The seam between the UI and the (to-be-implemented) AI backend.

Only the *contract* is fixed here. The implementation is intentionally left to
the backend author.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Optional

from openai import OpenAI
from pydantic import BaseModel, SecretStr

from chatapp.events import AIEvent, TextDelta, ToolCall, ToolResult
from chatapp.messages import Message

__all__ = ["AIClient"]



class AttributeDefinition(BaseModel):
    name: str
    type: str
    description: str


class AttributeAssignment(BaseModel):
    name: str
    value: str | int | bool

class GetHierarchiesArgs(BaseModel):
    pass



class CreateHierarchyArgs(BaseModel):
    name: str
    description: str
    additional_attributes: list[AttributeDefinition]


class GetEntriesArgs(BaseModel):
    hierarchy: str


class CreateEntryArgs(BaseModel):
    hierarchy: str
    name: str
    attributes: list[AttributeAssignment]
    body: Optional[str]


@dataclass(frozen=True)
class ToolOutcome:
    """The structured result of running a tool handler.

    ``content`` is the JSON payload handed back to the model (and surfaced in
    the :class:`~chatapp.events.ToolResult` event); ``is_error`` says whether
    the operation succeeded, so the UI and the model can distinguish a genuine
    failure from a normal result.
    """

    is_error: bool
    content: str


def _ok(payload: object) -> ToolOutcome:
    """A successful tool outcome carrying ``payload`` as JSON."""
    return ToolOutcome(is_error=False, content=json.dumps(payload))


def _fail(payload: object) -> ToolOutcome:
    """A failed tool outcome carrying ``payload`` as JSON."""
    return ToolOutcome(is_error=True, content=json.dumps(payload))


# Build the handlers:
def get_hierarchies(raw_args: str, config: AIClientConfig) -> ToolOutcome:
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

    return _ok(hierarchies)

def _safe_segment(name: str) -> str | None:
    cleaned = name.strip()
    if not cleaned or cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        return None
    return cleaned


def _now_iso() -> str:
    """Current local time as an ISO 8601 string with a timezone offset."""
    return datetime.now().astimezone().isoformat()


def _render_schema(args: CreateHierarchyArgs) -> str:
    lines = [f"# {args.name}", "", args.description.strip()]
    if args.additional_attributes:
        lines += ["", "## Attributes", ""]
        lines += [
            f"- **{attr.name}** (`{attr.type}`): {attr.description}"
            for attr in args.additional_attributes
        ]
    return "\n".join(lines).rstrip() + "\n"


def _render_entry(attributes: list[AttributeAssignment],body: str | None,) -> str:
    lines = ["---"]
    lines += [
        f"{attr.name}: {json.dumps(attr.value)}"
        for attr in attributes
        if attr.name != "recorded_at"
    ]
    lines.append(f"recorded_at: {json.dumps(_now_iso())}")
    lines.append("---")

    if body:
        lines.append('')
        lines.append(body)

    return "\n".join(lines) + "\n"


def create_hierarchy(raw_args: str, config: AIClientConfig) -> ToolOutcome:
    args = CreateHierarchyArgs.model_validate_json(raw_args or "{}")

    name = _safe_segment(args.name)
    if name is None:
        return _fail({"created": False, "error": f"invalid hierarchy name: {args.name!r}"})

    hierarchy_dir = config.data_dir / name
    if hierarchy_dir.exists():
        return _fail({"name": name, "created": False, "error": "hierarchy already exists"})

    hierarchy_dir.mkdir(parents=True)
    (hierarchy_dir / "SCHEMA.md").write_text(_render_schema(args), encoding="utf-8")

    return _ok({"name": name, "created": True})


def get_entries(raw_args: str, config: AIClientConfig) -> ToolOutcome:
    args = GetEntriesArgs.model_validate_json(raw_args or "{}")

    entries: list[dict[str, str]] = []
    hierarchy_dir = config.data_dir / args.hierarchy
    if hierarchy_dir.is_dir():
        for child in sorted(hierarchy_dir.iterdir()):
            if child.is_file() and child.suffix == ".md" and child.name != "SCHEMA.md":
                entries.append({"name": child.stem})

    return _ok(entries)


def create_entry(raw_args: str, config: AIClientConfig) -> ToolOutcome:
    args = CreateEntryArgs.model_validate_json(raw_args or "{}")

    hierarchy = _safe_segment(args.hierarchy)
    entry_name = _safe_segment(args.name.removesuffix(".md"))
    if hierarchy is None or entry_name is None:
        return _fail({"created": False, "error": "invalid hierarchy or entry name"})

    hierarchy_dir = config.data_dir / hierarchy
    if not hierarchy_dir.is_dir():
        return _fail({"created": False, "error": f"hierarchy {hierarchy!r} does not exist"})

    entry_path = hierarchy_dir / f"{entry_name}.md"
    if entry_path.exists():
        return _fail(
            {"hierarchy": hierarchy, "name": entry_name, "created": False,
             "error": "entry already exists"}
        )

    entry_path.write_text(_render_entry(args.attributes, args.body), encoding="utf-8")
    # if args.body:
    #     entry_path.write_text("\n", encoding="utf-8")
    #     entry_path.write_text(args.body, encoding="utf-8")

    return _ok({"hierarchy": hierarchy, "name": entry_name, "created": True})


def _extract_tool_calls(response: Any) -> list[Any]:
    return [x for x in response.output if x.type == "function_call"]

@dataclass(frozen=True)
class _Tool:

    description: str
    args_type: type[BaseModel]
    handler: Callable[[str, AIClientConfig], ToolOutcome]


TOOLS_REGISTER: dict[str, _Tool] = {
    "get_hierarchies": _Tool(
        description="Use this tool to retrieve all hierarchies.",
        args_type=GetHierarchiesArgs,
        handler=get_hierarchies
    ),
    "create_hierarchy" : _Tool(
        description=(
            "Create a new hierarchy. `additional_attributes` MUST be an array of "
            "OBJECTS, each with string fields `name`, `type`, and `description` — "
            "never an array of strings. "
            'Example: additional_attributes=[{"name": "location_type", '
            '"type": "string", "description": "Type of location"}].'
        ),
        args_type=CreateHierarchyArgs,
        handler=create_hierarchy
    ),
    "get_entries": _Tool(
        description="Use this tool to retrieve all entries in a given hierarchy.",
        args_type=GetEntriesArgs,
        handler=get_entries
    ),
    "create_entry": _Tool(
        description=(
            "Create a new entry within a hierarchy. `attributes` MUST be an array "
            "of OBJECTS, each with a `name` and a `value` (string or integer) — "
            'never strings like "title: Sofia". '
            'Example: attributes=[{"name": "title", "value": "Sofia"}, '
            '{"name": "source", "value": "stated"}].'
        ),
        args_type=CreateEntryArgs,
        handler=create_entry
    ),
}

def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a pydantic JSON schema with ``$ref``/``$defs`` inlined.

    Pydantic emits array-of-model fields as ``items: {"$ref": "#/$defs/..."}``.
    Gemini's function-call schema parser does not reliably follow ``$ref`` into
    ``$defs``; when it cannot resolve the reference it drops the object shape of
    the array items and the model sends arrays of bare strings instead. Inlining
    every definition makes each item's structure self-contained and visible, so
    the correct object shape is emitted on the first call.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = resolve(defs[ref.rsplit("/", 1)[-1]])
                siblings = {k: resolve(v) for k, v in node.items() if k != "$ref"}
                return {**target, **siblings}
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    resolved = resolve(schema)
    return resolved if isinstance(resolved, dict) else {}


AI_TOOLS: list[Any] = [
    {
        "type": "function",
        "name": tool_name,
        "description": tool_props.description,
        "parameters": {
            **_inline_defs(tool_props.args_type.model_json_schema()),
            "additionalProperties": False,
        },
        "strict": True,
    }
    for tool_name, tool_props in TOOLS_REGISTER.items()
]
print(AI_TOOLS)


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

    def _run_tool(self, name: str, raw_args: str) -> ToolOutcome:
        """Execute a registered tool, converting any failure into a result.

        Malformed model-generated arguments (a :class:`pydantic.ValidationError`
        raised while the handler parses ``raw_args``) and any other handler error
        are caught and returned as a :class:`ToolOutcome` with ``is_error=True``.
        The model receives the problem as tool output and can retry, rather than
        the exception escaping and crashing the stream.
        """
        tool = TOOLS_REGISTER.get(name)
        if tool is None:
            return _fail({"error": f"unknown tool: {name!r}"})
        try:
            return tool.handler(raw_args, self._config)
        except Exception as exc:  # returned to the model, never crashes the stream
            return _fail({"error": f"{type(exc).__name__}: {exc}"})

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

        # Each turn streams the model's output as it is produced. If the turn
        # ends in tool calls we run them, feed the results back, and stream the
        # next turn; a turn with no tool calls is the final answer, so we stop.
        while True:
            response: Any = None
            stream: Any = self._client.responses.create(
                model=self._config.model_name,
                input=context,
                tools=AI_TOOLS,
                stream=True,
            )
            for event in stream:
                # Emit text the moment it arrives, rather than buffering the
                # whole response.
                if event.type == "response.output_text.delta":
                    yield TextDelta(text=event.delta)
                elif event.type == "response.completed":
                    response = event.response

            if response is None:
                break

            tool_calls = _extract_tool_calls(response)

            print(f"Discovered {len(tool_calls)} tool calls:")
            for tc in tool_calls:
                print(tc)
            print()
            
            if not tool_calls:
                break

            context.extend(response.output)
            
            for tc in tool_calls:
                try:
                    parsed_args = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    parsed_args = {}
                yield ToolCall(
                    name=tc.name,
                    arguments=parsed_args if isinstance(parsed_args, dict) else {},
                )

                outcome = self._run_tool(tc.name, tc.arguments)

                yield ToolResult(
                    name=tc.name,
                    content=outcome.content,
                    is_error=outcome.is_error,
                )

                context.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc.call_id,
                        "output": outcome.content,
                    }
                )
