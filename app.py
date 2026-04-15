"""
Shared application state: FastMCP instance, T212 client, environment config.
All tool modules import from here — no circular imports.
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastmcp import FastMCP

from t212_client import T212Client

load_dotenv()

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

T212_API_KEY = os.environ["T212_API_KEY"]
T212_API_SECRET = os.environ["T212_API_SECRET"]
FINNHUB_API_KEY = os.environ["FINNHUB_API_KEY"]
T212_MODE = os.environ.get("T212_MODE", "demo")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")

# ---------------------------------------------------------------------------
# Shared state — populated by lifespan
# ---------------------------------------------------------------------------

t212: T212Client


@asynccontextmanager
async def lifespan(server):
    global t212
    t212 = T212Client(api_key=T212_API_KEY, api_secret=T212_API_SECRET, mode=T212_MODE)
    yield
    await t212.aclose()


mcp = FastMCP(
    "Investment Analyser",
    lifespan=lifespan,
)



