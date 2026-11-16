"""`python -m desk` / `uv run desk`: start the dashboard + scheduler on 127.0.0.1:8000."""
from __future__ import annotations

import logging
import sys

import uvicorn

from .config import ConfigError, load_settings


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        settings = load_settings()
        from .server import create_app
        app = create_app(settings)
    except (ConfigError, RuntimeError) as e:
        print(f"Agent Desk cannot start: {e}", file=sys.stderr)
        return 2
    print(f"Agent Desk [{settings.mode.upper()} mode] -> http://{settings.host}:{settings.port}")
    if settings.uses_llm:
        print(f"AI agents: {settings.llm_provider} (chief: {settings.model_chief}, analysts: "
              f"{settings.model_analysts})"
              f"{' - your Claude plan (no API bill; plan usage limits apply)' if settings.llm_provider == 'claude_code' else ' - free' if settings.llm_is_free else ''}")
    else:
        print("AI agents: scripted (offline demo)")
    if settings.phone_access:
        urls = getattr(app.state, "phone_urls", [])
        key = getattr(app.state, "access_key", "")
        if urls:
            print(f"Phone (same Wi-Fi), open once in Safari: {urls[0]}?key={key}")
            print("  then Share > Add to Home Screen. Keep this link private: it unlocks the dashboard controls.")
        else:
            print("Phone access is on, but no Wi-Fi address was found.")
    sys.stdout.flush()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
