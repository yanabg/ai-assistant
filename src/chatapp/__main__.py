"""Entry point: build the config, wire the pieces together, run the UI."""

from __future__ import annotations

import sys
import tkinter as tk

from pydantic import ValidationError

from chatapp.ai_client import AIClient
from chatapp.app import ChatApp
from chatapp.async_bridge import AsyncLoopThread
from chatapp.config import Config, load_config


def _report_config_error(error: ValidationError) -> None:
    missing = ", ".join(str(item["loc"][0]) for item in error.errors() if item.get("loc"))
    sys.stderr.write(
        "Configuration error: could not build Config"
        + (f" (problem with: {missing})" if missing else "")
        + ".\n"
        "Copy .env.example to .env and set CHATAPP_API_KEY (and adjust the "
        "other CHATAPP_* values as needed), then try again.\n"
    )


def main() -> None:
    """Compose the application and enter the tkinter main loop."""
    try:
        config: Config = load_config()
    except ValidationError as error:
        _report_config_error(error)
        raise SystemExit(1) from error

    loop = AsyncLoopThread()
    loop.start()
    try:
        root = tk.Tk()
        client = AIClient(config.ai_client_config())
        ChatApp(root, client=client, loop=loop, model_name=config.model_name)
        root.mainloop()
    finally:
        loop.stop()


if __name__ == "__main__":
    main()
