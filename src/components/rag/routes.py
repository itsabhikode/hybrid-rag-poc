from fastapi import APIRouter
from src.components.rag.routers.get_indicator_type import router as get_indicator_type_router
from src.components.rag.routers.get_full_context_of_indicator import router as get_full_context_of_indicator_router
from src.components.rag.routers.graph_traversal_connected import router as graph_traversal_connected_router
from src.components.rag.routers.search import router as search_router

# Create main router
router = APIRouter()


# Include all routers
router.include_router(search_router)  # Main search endpoint
router.include_router(get_indicator_type_router)
router.include_router(get_full_context_of_indicator_router)
router.include_router(graph_traversal_connected_router)
