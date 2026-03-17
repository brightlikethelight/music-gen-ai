"""
Simple web server for MusicGen UI.
No frameworks, just what we need.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from musicgen import __version__

# Get static directory
STATIC_DIR = Path(__file__).parent.parent / "static"
if not STATIC_DIR.exists():
    # Try alternative location
    STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    """Create web application."""
    app = FastAPI(
        title="MusicGen Web UI",
        description="Simple web interface for music generation",
        version=__version__,
    )

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Also mount the API app
    from ..api.rest.app import app as api_app

    app.mount("/api", api_app)

    _FALLBACK_HTML = """\
<!DOCTYPE html>
<html>
<head><title>MusicGen</title></head>
<body>
<h1>MusicGen Web UI</h1>
<p>Static files not found. API available at <a href="/api/docs">/api/docs</a></p>
</body>
</html>"""

    @app.get("/", response_class=HTMLResponse)
    async def root() -> Response:
        """Serve main page."""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return HTMLResponse(_FALLBACK_HTML)

    return app


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the web server."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """Main entry point for web server."""
    run_server()


if __name__ == "__main__":
    main()
