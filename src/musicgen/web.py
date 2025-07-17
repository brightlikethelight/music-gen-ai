"""
Simple web server for MusicGen UI.
No frameworks, just what we need.
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

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
        version="2.0.0"
    )
    
    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    
    # Also mount the API app
    from .api import app as api_app
    app.mount("/api", api_app)
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve main page."""
        index_path = STATIC_DIR / "index.html"
        
        if index_path.exists():
            return FileResponse(index_path)
        else:
            # Fallback HTML if static files not found
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>MusicGen</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        max-width: 800px;
                        margin: 50px auto;
                        padding: 20px;
                    }
                    h1 { color: #333; }
                    .error {
                        background: #fee;
                        border: 1px solid #fcc;
                        padding: 20px;
                        border-radius: 5px;
                        color: #c00;
                    }
                </style>
            </head>
            <body>
                <h1>MusicGen Web UI</h1>
                <div class="error">
                    <h2>Static files not found</h2>
                    <p>The web UI files could not be loaded.</p>
                    <p>API is still available at <a href="/api/docs">/api/docs</a></p>
                </div>
            </body>
            </html>
            """
    
    return app


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the web server."""
    import uvicorn
    
    app = create_app()
    uvicorn.run(app, host=host, port=port)


def main():
    """Main entry point for web server."""
    run_server()


if __name__ == "__main__":
    main()