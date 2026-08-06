# desktop-chat

A streaming desktop chat client built with the Python standard-library
`tkinter`/`ttk` toolkit. This repository is the **foundation**: the UI, the
configuration, and the async/streaming plumbing are complete. The AI backend is
a single, clearly-marked seam (`AIClient.send`) that is intentionally left
unimplemented for you to fill in.

## Requirements

- Python **3.11+**
- [`uv`](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
uv sync --all-groups        # create the venv and install all dependency groups
cp .env.example .env        # then edit .env and set CHATAPP_API_KEY
```

## Run

```bash
uv run chatapp
# or
uv run python -m chatapp
```

Until you implement `AIClient.send`, sending a message surfaces a friendly
"client not implemented" notice as an error message in the transcript. This is
deliberate: it exercises the full async → UI error path without a backend.

## Develop

```bash
uv run ruff format .        # format
uv run ruff check .         # lint
uv run mypy src             # type-check (strict)
```

## Where the seam is

Implement your backend in `src/chatapp/ai_client.py`:

```python
class AIClient:
    def send(
        self,
        history: Sequence[Message],
        user_message: str,
    ) -> AsyncIterator[AIEvent]: ...
```

`send` returns an **async iterator** of typed events
(`TextDelta` | `ToolCall`, discriminated on `type`). Yield events as they are
produced; the UI streams them incrementally and never blocks. Construction
receives configuration explicitly (`AIClientConfig`) — no global lookups.

<!-- ## Layout - DELETE THIS, NOT NEEDED

```
src/chatapp/
├── __main__.py       # entry point: build Config, wire everything, run mainloop
├── config.py         # single pydantic-settings Config (the only env reader)
├── messages.py       # Message model (conversation history)
├── events.py         # TextDelta / ToolCall discriminated union
├── ai_client.py      # the AIClient seam (contract only)
├── async_bridge.py   # asyncio event loop on a background thread
├── theme.py          # palette, spacing, font resolution
└── app.py            # the tkinter UI + streaming state machine
``` -->
