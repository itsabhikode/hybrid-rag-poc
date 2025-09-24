from hypercorn.asyncio import serve
from hypercorn.config import Config
from src.main import app
import asyncio

if __name__ == "__main__":
    config = Config()
    config.bind = ["0.0.0.0:8000"]
    config.use_reloader = True
    config.forwarded_allow_ips = "*"
    asyncio.run(serve(app, config))