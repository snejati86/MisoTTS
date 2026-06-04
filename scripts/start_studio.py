from __future__ import annotations

import argparse
import os
import platform
import threading
import time
import webbrowser

import uvicorn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MisoTTS Studio frontend and backend.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    parser.add_argument("--autoload", action="store_true", help="Warm the model during server startup.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the Studio URL in a browser.")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"))
    return parser.parse_args()


def _open_browser(url: str) -> None:
    # Give Uvicorn a moment to bind the socket before opening the browser.
    time.sleep(1.2)
    webbrowser.open(url)


def main() -> None:
    args = _parse_args()

    os.environ.setdefault("MISO_TTS_AUTOLOAD", "1" if args.autoload else "0")
    os.environ.setdefault("NO_TORCH_COMPILE", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    public_url = f"http://127.0.0.1:{args.port}" if args.host in {"0.0.0.0", "::"} else f"http://{args.host}:{args.port}"
    print("Starting MisoTTS Studio")
    print(f"  URL: {public_url}")
    print(f"  Autoload: {os.environ['MISO_TTS_AUTOLOAD']}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print("  Frontend and backend are served by the same FastAPI app.")

    if not args.no_browser and args.host in {"127.0.0.1", "localhost"}:
        threading.Thread(target=_open_browser, args=(public_url,), daemon=True).start()

    uvicorn.run(
        "studio_server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
