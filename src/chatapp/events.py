"""Typed streaming events emitted by :class:`~chatapp.ai_client.AIClient`.

The events form a discriminated union keyed on the literal ``type`` field, so a
consumer can ``match`` (or ``isinstance``-narrow) exhaustively.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["AIEvent", "StreamError", "TextDelta", "ToolCall", "ToolResult"]


class TextDelta(BaseModel):
    """An incremental chunk of assistant text produced while streaming."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolCall(BaseModel):
    """A request from the model to invoke a named tool."""

    type: Literal["tool_call"] = "tool_call"
    name: str
    arguments: Mapping[str, object] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of a previously announced :class:`ToolCall`."""

    type: Literal["tool_result"] = "tool_result"
    name: str
    content: str
    is_error: bool = False


class StreamError(BaseModel):
    """A failure the client wants rendered as a message rather than raised.

    Raising out of the stream is also supported and lands in the same UI state;
    use this event when the stream can report the problem and keep going.
    """

    type: Literal["error"] = "error"
    message: str
    retryable: bool = False


AIEvent = TextDelta | ToolCall | ToolResult | StreamError
"""Every event the client can stream. Discriminate on the ``type`` field."""
