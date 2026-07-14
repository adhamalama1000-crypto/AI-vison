"""
Entrypoint.

Run directly:      python run.py
Or with uvicorn:   uvicorn run:app --host 0.0.0.0 --port 8000
"""

from rtsp_backend.app import build_app
from rtsp_backend.config import load_settings

settings = load_settings()
app = build_app(settings)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
