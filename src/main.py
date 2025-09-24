from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.setup_routes import setup_routes
from src.shared.utils.settings import get_settings

settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="Document Intelligence Pipeline API",
    description="API for managing document intelligence pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
setup_routes(app)
