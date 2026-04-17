import asyncio
import base64
import httpx


class T212Client:
    """Async HTTP client for the Trading 212 REST API."""

    _BASE_URLS = {
        "demo": "https://demo.trading212.com/api/v0",
        "live": "https://live.trading212.com/api/v0",
    }

    # Max concurrent in-flight T212 requests across all tools.
    # T212 allows ~100 req/min; 2 concurrent prevents burst 429s while
    # still letting get_portfolio + get_account_summary run in parallel.
    _CONCURRENCY = 2

    def __init__(self, api_key: str, api_secret: str, mode: str = "demo"):
        if mode not in self._BASE_URLS:
            raise ValueError(f"T212_MODE must be 'demo' or 'live', got: {mode!r}")
        base_url = self._BASE_URLS[mode]
        credentials = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Basic {credentials}"},
            timeout=15.0,
        )
        self.mode = mode
        self._sem = asyncio.Semaphore(self._CONCURRENCY)

    # ── Internal Retry ───────────────────────────────────────────────────────

    async def _get_with_retry(self, path: str, params: dict = None) -> httpx.Response:
        for attempt in range(4):
            async with self._sem:                  # hold lock only for the HTTP call
                resp = await self._client.get(path, params=params)
            if resp.status_code == 429:
                if attempt == 3:
                    self._raise_for_status(resp)
                try:
                    retry_after = int(resp.headers.get("Retry-After", "5"))
                except ValueError:
                    retry_after = 5
                wait_time = max(retry_after, 2 ** attempt)
                await asyncio.sleep(wait_time)     # sleep outside the lock
            else:
                self._raise_for_status(resp)
                return resp
        return resp

    # ── Portfolio & Account ──────────────────────────────────────────────────

    async def get_portfolio(self) -> list[dict]:
        """GET /equity/portfolio — open positions."""
        resp = await self._get_with_retry("/equity/portfolio")
        return resp.json()

    async def get_account_summary(self) -> dict:
        """GET /equity/account/summary — account totals."""
        resp = await self._get_with_retry("/equity/account/summary")
        return resp.json()

    # ── History ──────────────────────────────────────────────────────────────

    async def get_order_history(self, limit: int = 50) -> list[dict]:
        """GET /equity/history/orders — recent order history."""
        resp = await self._get_with_retry("/equity/history/orders", params={"limit": limit})
        data = resp.json()
        return data.get("items", data) if isinstance(data, dict) else data

    async def get_dividend_history(self, limit: int = 20) -> list[dict]:
        """GET /equity/history/dividends — dividend payments received."""
        resp = await self._get_with_retry("/equity/history/dividends", params={"limit": limit})
        data = resp.json()
        return data.get("items", data) if isinstance(data, dict) else data

    async def get_transaction_history(self, limit: int = 20) -> list[dict]:
        """GET /equity/history/transactions — cash transactions (deposits, withdrawals)."""
        resp = await self._get_with_retry("/equity/history/transactions", params={"limit": limit})
        data = resp.json()
        return data.get("items", data) if isinstance(data, dict) else data

    # ── Orders ───────────────────────────────────────────────────────────────

    async def get_open_orders(self) -> list[dict]:
        """GET /equity/orders — currently open/pending orders."""
        resp = await self._get_with_retry("/equity/orders")
        data = resp.json()
        return data.get("items", data) if isinstance(data, dict) else data

    # ── Pies ─────────────────────────────────────────────────────────────────

    async def get_pies(self) -> list[dict]:
        """GET /equity/pies — all pies (automated portfolios)."""
        resp = await self._get_with_retry("/equity/pies")
        return resp.json()

    async def get_pie(self, pie_id: int) -> dict:
        """GET /equity/pies/{id} — detailed breakdown of a single pie."""
        resp = await self._get_with_retry(f"/equity/pies/{pie_id}")
        return resp.json()

    # ── Metadata ─────────────────────────────────────────────────────────────

    async def find_instrument(self, query: str) -> dict | None:
        """Search instruments by shortName (ticker) or name. Returns first match."""
        resp = await self._get_with_retry("/equity/metadata/instruments")
        instruments = resp.json()
        q = query.upper()
        for inst in instruments:
            if inst.get("shortName", "").upper() == q:
                return inst
        for inst in instruments:
            if q in inst.get("name", "").upper():
                return inst
        return None

    # ── Shared ───────────────────────────────────────────────────────────────

    async def aclose(self):
        await self._client.aclose()

    @staticmethod
    def _raise_for_status(resp: httpx.Response):
        if resp.status_code == 401:
            raise PermissionError("T212 authentication failed — check T212_API_KEY")
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                raise RuntimeError(f"T212 rate limit exceeded — you MUST wait at least {retry_after} seconds before retrying.")
            else:
                raise RuntimeError("T212 rate limit exceeded — try again in a moment")
        if resp.status_code >= 500:
            raise RuntimeError(f"T212 API error ({resp.status_code}) — service may be unavailable")
        resp.raise_for_status()
