"""Serve Kev's System One HTTP API (mock backend in v0.1)."""

from __future__ import annotations

import uvicorn

from kev.server import create_app


def main() -> None:
    uvicorn.run(create_app(model="mock"), host="0.0.0.0", port=8787, reload=False)


if __name__ == "__main__":
    main()
