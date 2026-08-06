"""The tkinter chat UI.

Owns every widget and the streaming state machine. Streaming work happens on a
background asyncio loop (:mod:`chatapp.async_bridge`); results arrive on a
thread-safe queue that this widget polls with ``after``, so the UI never blocks
while a long response streams in.
"""

from __future__ import annotations

import asyncio
import queue
import tkinter as tk
import tkinter.font as tkfont
from concurrent.futures import Future
from dataclasses import dataclass
from tkinter import ttk
from typing import assert_never

from chatapp.ai_client import AIClient
from chatapp.async_bridge import AsyncLoopThread
from chatapp.events import AIEvent, StreamError, TextDelta, ToolCall, ToolResult
from chatapp.messages import Message
from chatapp.theme import PALETTE, SPACING, resolve_fonts

# --- Messages passed from the loop thread back to the UI thread ---------------


@dataclass(frozen=True)
class _Event:
    event: AIEvent


@dataclass(frozen=True)
class _Error:
    error: BaseException


@dataclass(frozen=True)
class _Done:
    pass


@dataclass(frozen=True)
class _Cancelled:
    pass


_StreamMessage = _Event | _Error | _Done | _Cancelled


# --- The application widget ---------------------------------------------------


class ChatApp:
    """The whole chat window, wired to an :class:`AIClient` and a loop thread."""

    _DRAIN_MS = 33
    _BODY_SIZE = 12
    _META_SIZE = 10
    _MIN_INPUT_LINES = 1
    _MAX_INPUT_LINES = 7

    def __init__(
        self,
        root: tk.Tk,
        *,
        client: AIClient,
        loop: AsyncLoopThread,
        model_name: str,
    ) -> None:
        self._root = root
        self._client = client
        self._loop = loop
        self._model_name = model_name

        self._history: list[Message] = []
        self._queue: queue.Queue[_StreamMessage] = queue.Queue()
        self._future: Future[None] | None = None
        self._drain_job: str | None = None

        self._streaming = False
        self._assistant_open = False
        self._assistant_buffer = ""

        self._build_fonts()
        self._build_style()
        self._build_layout()
        self._configure_tags()
        self._show_placeholder()

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._input.focus_set()
        self._drain_job = self._root.after(self._DRAIN_MS, self._drain_queue)

    # -- Construction ---------------------------------------------------------

    def _build_fonts(self) -> None:
        ui_family, mono_family = resolve_fonts(self._root)
        self._font_body = tkfont.Font(root=self._root, family=ui_family, size=self._BODY_SIZE)
        self._font_meta = tkfont.Font(root=self._root, family=ui_family, size=self._META_SIZE)
        self._font_title = tkfont.Font(
            root=self._root, family=ui_family, size=self._BODY_SIZE + 2, weight="bold"
        )
        self._font_button = tkfont.Font(
            root=self._root, family=ui_family, size=self._BODY_SIZE - 1, weight="bold"
        )
        self._font_mono = tkfont.Font(root=self._root, family=mono_family, size=self._BODY_SIZE - 1)

    def _build_style(self) -> None:
        self._root.configure(background=PALETTE.window)
        self._root.title("Chat")
        self._root.minsize(460, 420)
        self._root.geometry("720x680")

        style = ttk.Style(self._root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=PALETTE.window)
        style.configure("Surface.TFrame", background=PALETTE.surface)
        style.configure(
            "Title.TLabel",
            background=PALETTE.window,
            foreground=PALETTE.text_primary,
            font=self._font_title,
        )
        style.configure(
            "Subtitle.TLabel",
            background=PALETTE.window,
            foreground=PALETTE.text_muted,
            font=self._font_meta,
        )
        style.configure("Vertical.TScrollbar", background=PALETTE.window, borderwidth=0)

        style.configure(
            "Accent.TButton",
            font=self._font_button,
            foreground=PALETTE.user_fg,
            background=PALETTE.accent,
            borderwidth=0,
            focuscolor=PALETTE.accent,
            padding=(18, 9),
        )
        style.map(
            "Accent.TButton",
            background=[("disabled", PALETTE.disabled_bg), ("active", PALETTE.accent_active)],
            foreground=[("disabled", PALETTE.disabled_fg)],
        )
        style.configure(
            "Ghost.TButton",
            font=self._font_button,
            foreground=PALETTE.text_primary,
            background=PALETTE.ghost_bg,
            borderwidth=0,
            focuscolor=PALETTE.ghost_bg,
            padding=(18, 9),
        )
        style.map(
            "Ghost.TButton",
            background=[("disabled", PALETTE.disabled_bg), ("active", PALETTE.ghost_active)],
            foreground=[("disabled", PALETTE.disabled_fg)],
        )

    def _build_layout(self) -> None:
        outer = ttk.Frame(self._root, style="App.TFrame", padding=SPACING.gutter)
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        header = ttk.Frame(outer, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, SPACING.gutter))
        ttk.Label(header, text="Chat", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text=f"model · {self._model_name}", style="Subtitle.TLabel").pack(
            anchor="w", pady=(2, 0)
        )

        # Transcript (scrollable, selectable, read-only).
        transcript_wrap = tk.Frame(
            outer,
            background=PALETTE.surface,
            highlightthickness=1,
            highlightbackground=PALETTE.border,
            highlightcolor=PALETTE.border,
        )
        transcript_wrap.grid(row=1, column=0, sticky="nsew")
        transcript_wrap.rowconfigure(0, weight=1)
        transcript_wrap.columnconfigure(0, weight=1)

        self._transcript = tk.Text(
            transcript_wrap,
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
            padx=SPACING.transcript_padx,
            pady=SPACING.transcript_pady,
            background=PALETTE.surface,
            foreground=PALETTE.text_primary,
            font=self._font_body,
            cursor="arrow",
            insertwidth=0,
            state="disabled",
            takefocus=False,
        )
        self._transcript.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(
            transcript_wrap, orient="vertical", command=self._transcript.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._transcript.configure(yscrollcommand=scrollbar.set)

        # Composer.
        composer = ttk.Frame(outer, style="App.TFrame")
        composer.grid(row=2, column=0, sticky="ew", pady=(SPACING.gutter, 0))
        composer.columnconfigure(0, weight=1)

        input_wrap = tk.Frame(
            composer,
            background=PALETTE.surface,
            highlightthickness=1,
            highlightbackground=PALETTE.border,
            highlightcolor=PALETTE.accent,
        )
        input_wrap.grid(row=0, column=0, sticky="ew")
        input_wrap.columnconfigure(0, weight=1)

        self._input = tk.Text(
            input_wrap,
            height=self._MIN_INPUT_LINES,
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
            padx=12,
            pady=9,
            background=PALETTE.surface,
            foreground=PALETTE.text_primary,
            insertbackground=PALETTE.text_primary,
            font=self._font_body,
        )
        self._input.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        self._input.bind("<Return>", self._on_return)
        self._input.bind("<Shift-Return>", self._on_shift_return)
        self._input.bind("<KeyRelease>", self._autosize_input)

        buttons = ttk.Frame(composer, style="App.TFrame")
        buttons.grid(row=0, column=1, sticky="ns", padx=(SPACING.gutter, 0))
        self._send_btn = ttk.Button(
            buttons, text="Send", style="Accent.TButton", command=self._on_send
        )
        self._send_btn.pack(fill="x")
        self._stop_btn = ttk.Button(
            buttons, text="Stop", style="Ghost.TButton", command=self._on_stop
        )
        self._stop_btn.pack(fill="x", pady=(SPACING.bubble_gap, 0))
        self._stop_btn.state(["disabled"])

    def _configure_tags(self) -> None:
        t = self._transcript
        t.tag_configure(
            "user",
            justify="right",
            lmargin1=SPACING.bubble_inset,
            lmargin2=SPACING.bubble_inset,
            rmargin=4,
            background=PALETTE.user_bg,
            foreground=PALETTE.user_fg,
            spacing1=SPACING.bubble_gap,
            spacing3=SPACING.bubble_gap,
        )
        t.tag_configure(
            "assistant",
            justify="left",
            lmargin1=4,
            lmargin2=4,
            rmargin=SPACING.bubble_inset,
            background=PALETTE.assistant_bg,
            foreground=PALETTE.assistant_fg,
            spacing1=SPACING.bubble_gap,
            spacing3=SPACING.bubble_gap,
        )
        t.tag_configure(
            "error",
            justify="left",
            lmargin1=4,
            lmargin2=4,
            rmargin=SPACING.bubble_inset,
            background=PALETTE.error_bg,
            foreground=PALETTE.error_fg,
            spacing1=SPACING.bubble_gap,
            spacing3=SPACING.bubble_gap,
        )
        t.tag_configure(
            "tool",
            justify="left",
            lmargin1=4,
            lmargin2=4,
            rmargin=SPACING.bubble_inset,
            foreground=PALETTE.tool_fg,
            font=self._font_mono,
            spacing1=SPACING.bubble_gap // 2,
            spacing3=SPACING.bubble_gap // 2,
        )
        t.tag_configure(
            "meta",
            justify="center",
            foreground=PALETTE.text_muted,
            font=self._font_meta,
            spacing1=SPACING.bubble_gap,
            spacing3=SPACING.bubble_gap,
        )

    def _show_placeholder(self) -> None:
        self._emit_inline("Type a message and press Enter to begin.", "meta")

    # -- Transcript rendering primitives --------------------------------------

    def _at_bottom(self) -> bool:
        return self._transcript.yview()[1] >= 0.999

    def _emit_inline(self, text: str, tag: str) -> None:
        """Append ``text`` (which may contain newlines) at the end under ``tag``.

        Line breaks inside ``text`` are inserted *untagged* so a coloured tag
        hugs the glyphs instead of stretching to the window edge. Consecutive
        calls concatenate, which is what streaming deltas need.
        """
        stick = self._at_bottom()
        self._transcript.configure(state="normal")
        for index, part in enumerate(text.split("\n")):
            if index > 0:
                self._transcript.insert("end", "\n")
            if part:
                self._transcript.insert("end", part, (tag,))
        self._transcript.configure(state="disabled")
        if stick:
            self._transcript.see("end")

    def _separate(self) -> None:
        """Terminate the current line so the next block starts fresh."""
        if self._transcript.index("end-1c") == "1.0":
            return
        if self._transcript.get("end-2c", "end-1c") == "\n":
            return
        self._transcript.configure(state="normal")
        self._transcript.insert("end", "\n")
        self._transcript.configure(state="disabled")

    def _clear_placeholder_if_needed(self) -> None:
        if not self._history and not self._assistant_open:
            self._transcript.configure(state="normal")
            self._transcript.delete("1.0", "end")
            self._transcript.configure(state="disabled")

    # -- Composer behaviour ---------------------------------------------------

    def _on_return(self, _event: tk.Event[tk.Text]) -> str:
        self._on_send()
        return "break"

    def _on_shift_return(self, _event: tk.Event[tk.Text]) -> str:
        self._input.insert("insert", "\n")
        self._autosize_input()
        self._input.see("insert")
        return "break"

    def _autosize_input(self, _event: tk.Event[tk.Text] | None = None) -> None:
        line_count = int(self._input.index("end-1c").split(".")[0])
        target = max(self._MIN_INPUT_LINES, min(self._MAX_INPUT_LINES, line_count))
        if target != int(self._input.cget("height")):
            self._input.configure(height=target)

    def _set_streaming(self, streaming: bool) -> None:
        self._streaming = streaming
        if streaming:
            self._send_btn.state(["disabled"])
            self._stop_btn.state(["!disabled"])
            self._input.configure(state="disabled")
        else:
            self._send_btn.state(["!disabled"])
            self._stop_btn.state(["disabled"])
            self._input.configure(state="normal")
            self._input.focus_set()

    # -- Send / stop ----------------------------------------------------------

    def _on_send(self) -> None:
        if self._streaming:
            return
        text = self._input.get("1.0", "end-1c").strip()
        if not text:
            return

        self._input.delete("1.0", "end")
        self._autosize_input()

        self._clear_placeholder_if_needed()
        history_snapshot = tuple(self._history)

        self._separate()
        self._emit_inline(text, "user")
        self._history.append(Message(role="user", content=text))

        self._begin_assistant()
        self._set_streaming(True)
        self._future = self._loop.submit(self._consume(history_snapshot, text))

    def _on_stop(self) -> None:
        future = self._future
        if future is not None:
            future.cancel()
        self._stop_btn.state(["disabled"])

    def _begin_assistant(self) -> None:
        self._assistant_open = True
        self._assistant_buffer = ""
        self._separate()

    # -- Coroutine running on the loop thread ---------------------------------

    async def _consume(self, history: tuple[Message, ...], user_message: str) -> None:
        try:
            stream = self._client.send(history, user_message)
            async for event in stream:
                self._queue.put(_Event(event))
        except asyncio.CancelledError:
            # Raised when the user hits Stop; report it, then honour the cancel.
            self._queue.put(_Cancelled())
            raise
        except Exception as exc:  # surfaced as an error bubble, never a traceback
            self._queue.put(_Error(exc))
        else:
            self._queue.put(_Done())

    # -- Queue draining on the UI thread --------------------------------------

    def _drain_queue(self) -> None:
        try:
            while True:
                self._dispatch(self._queue.get_nowait())
        except queue.Empty:
            pass
        self._drain_job = self._root.after(self._DRAIN_MS, self._drain_queue)

    def _dispatch(self, message: _StreamMessage) -> None:
        match message:
            case _Event(event=event):
                self._handle_event(event)
            case _Error(error=error):
                self._render_error(error)
                self._finish_stream()
            case _Done():
                self._finish_stream()
            case _Cancelled():
                self._finish_stream(cancelled=True)
            case _:  # pragma: no cover - exhaustiveness guard
                assert_never(message)

    def _handle_event(self, event: AIEvent) -> None:
        if isinstance(event, TextDelta):
            self._assistant_buffer += event.text
            self._emit_inline(event.text, "assistant")
        elif isinstance(event, ToolCall):
            self._render_tool_call(event)
        elif isinstance(event, ToolResult):
            self._render_tool_result(event)
        elif isinstance(event, StreamError):
            self._render_stream_error(event)
        else:  # pragma: no cover - exhaustiveness guard
            assert_never(event)

    def _finish_stream(self, *, cancelled: bool = False) -> None:
        if not self._streaming:
            return
        if cancelled and self._assistant_open:
            self._emit_inline(" (stopped)", "meta")
        if self._assistant_buffer:
            self._history.append(Message(role="assistant", content=self._assistant_buffer))
        self._assistant_buffer = ""
        self._assistant_open = False
        self._future = None
        self._set_streaming(False)

    # -- Error / tool rendering -----------------------------------------------

    def _render_error(self, error: BaseException) -> None:
        self._separate()
        self._emit_inline(self._humanize_error(error), "error")

    @staticmethod
    def _humanize_error(error: BaseException) -> str:
        if isinstance(error, NotImplementedError):
            return (
                "⚠  The AI client isn't implemented yet. Fill in "
                "AIClient.send in src/chatapp/ai_client.py to start chatting."
            )
        detail = str(error).strip()
        name = type(error).__name__
        return f"⚠  {name}: {detail}" if detail else f"⚠  {name}"

    def _render_tool_call(self, call: ToolCall) -> None:
        self._separate()
        if call.arguments:
            rendered = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
            text = f"⚙  tool call → {call.name}({rendered})"
        else:
            text = f"⚙  tool call → {call.name}()"
        self._emit_inline(text, "tool")

    def _render_tool_result(self, result: ToolResult) -> None:
        self._separate()
        status = "error" if result.is_error else "result"
        text = f"⚙  tool {status} ← {result.name}: {result.content}"
        self._emit_inline(text, "error" if result.is_error else "tool")

    def _render_stream_error(self, error: StreamError) -> None:
        self._separate()
        retry_text = " (retryable)" if error.retryable else ""
        self._emit_inline(f"⚠  {error.message}{retry_text}", "error")

    # -- Shutdown -------------------------------------------------------------

    def _on_close(self) -> None:
        if self._future is not None:
            self._future.cancel()
        if self._drain_job is not None:
            self._root.after_cancel(self._drain_job)
            self._drain_job = None
        self._loop.stop()
        self._root.destroy()
