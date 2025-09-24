from fastapi import FastAPI

# Import dashboard routes
from src.components.rag.routes import router as rag_router


def setup_routes(app: FastAPI):
    """
    Set up all application routes by including routers from different modules.

    Args:
        app: FastAPI application instance
    """
    # Include dashboard routes
    app.include_router(rag_router)
    