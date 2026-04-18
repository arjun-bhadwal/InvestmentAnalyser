"""Tests for multi-currency stock detection in resolver._detect_unit_scale().

Covers:
- LSE stocks priced in USD (e.g. COPX.L)
- LSE stocks priced in pence (GBX) — yfinance "GBX" variant
- LSE stocks priced in pence (GBX) — yfinance "GBp" variant
- LSE stocks mislabelled as GBP but price > 1000 (SGLN.L pattern)
- Genuine GBP LSE stocks
- Non-LSE tickers (SAP.DE, LYI.F) using yfinance currency
- yfinance returning empty currency → suffix-map fallback
- yfinance raising an exception → suffix-map fallback
- T212 currencyCode="GBX" path in position_value()

yfinance pence variants: "GBX" (uppercase) or "GBp" (mixed-case) — both handled.
T212 pence variant: "GBX" uppercase — handled separately in helpers.position_value.
"""
import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out modules that are not installed in the test environment so that
# importing resolver works without the full dependency tree.
# ---------------------------------------------------------------------------
def _stub_module(name: str) -> MagicMock:
    m = MagicMock()
    m.__name__ = name
    m.__spec__ = types.ModuleType(name)
    return m

for _mod in ("yfinance", "cachetools", "cachetools.keys"):
    if _mod not in sys.modules:
        sys.modules[_mod] = _stub_module(_mod)

# cachetools.TTLCache stub that behaves like a plain dict (no TTL enforced in tests)
import cachetools as _ct
_ct.TTLCache = lambda maxsize, ttl: {}

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import resolver
from resolver import _detect_unit_scale, _clear_caches_for_tests

# Re-alias so patches land on the right object (resolver.yf was bound at import)
import yfinance as _yf_stub


def _make_fast_info(currency: str, last_price: float) -> MagicMock:
    fi = MagicMock()
    fi.currency = currency
    fi.last_price = last_price
    return fi


def _mock_ticker(currency: str, last_price: float) -> MagicMock:
    fi = _make_fast_info(currency, last_price)
    t = MagicMock()
    t.fast_info = fi
    return t


class TestDetectUnitScale(unittest.TestCase):

    def setUp(self):
        _clear_caches_for_tests()

    def tearDown(self):
        _clear_caches_for_tests()

    def _patch_ticker(self, currency: str, last_price: float):
        return patch.object(resolver.yf, "Ticker", return_value=_mock_ticker(currency, last_price))

    # --- LSE / .L tickers ---

    def test_copx_l_usd(self):
        """COPX.L is priced in USD on the LSE — must NOT be assigned GBP."""
        with self._patch_ticker("USD", 47.80):
            scale, ccy = _detect_unit_scale("COPX.L")
        self.assertEqual(scale, 1.0)
        self.assertEqual(ccy, "USD")

    def test_lloy_l_gbx_uppercase(self):
        """yfinance returns 'GBX' (uppercase) for pence-priced LSE stocks."""
        with self._patch_ticker("GBX", 5050.0):
            scale, ccy = _detect_unit_scale("LLOY.L")
        self.assertEqual(scale, 0.01)
        self.assertEqual(ccy, "GBP")

    def test_vwrp_l_gbp_mixedcase(self):
        """yfinance sometimes returns 'GBp' (mixed-case) for pence-priced stocks."""
        with self._patch_ticker("GBp", 3200.0):
            scale, ccy = _detect_unit_scale("VWRP.L")
        self.assertEqual(scale, 0.01)
        self.assertEqual(ccy, "GBP")

    def test_sgln_l_mislabelled_gbx(self):
        """Yahoo labels SGLN.L as GBP but price > 1000 → treat as pence."""
        with self._patch_ticker("GBP", 1850.0):
            scale, ccy = _detect_unit_scale("SGLN.L")
        self.assertEqual(scale, 0.01)
        self.assertEqual(ccy, "GBP")

    def test_genuine_gbp_low_price(self):
        """Standard GBP stock with price well under 1000 → no scaling."""
        with self._patch_ticker("GBP", 450.0):
            scale, ccy = _detect_unit_scale("BP.L")
        self.assertEqual(scale, 1.0)
        self.assertEqual(ccy, "GBP")

    def test_lse_yfinance_empty_currency_fallback(self):
        """yfinance returns empty currency for an LSE stock → fall back to GBP."""
        with self._patch_ticker("", 0.0):
            scale, ccy = _detect_unit_scale("RARE.L")
        self.assertEqual(scale, 1.0)
        self.assertEqual(ccy, "GBP")

    def test_lse_yfinance_exception_fallback(self):
        """yfinance raises an exception → fall back gracefully to GBP."""
        with patch.object(resolver.yf, "Ticker", side_effect=RuntimeError("timeout")):
            scale, ccy = _detect_unit_scale("ANY.L")
        self.assertEqual(scale, 1.0)
        self.assertEqual(ccy, "GBP")

    # --- Non-LSE tickers ---

    def test_sap_de_eur(self):
        """SAP.DE is priced in EUR — yfinance confirms, scale is 1.0."""
        with self._patch_ticker("EUR", 210.0):
            scale, ccy = _detect_unit_scale("SAP.DE")
        self.assertEqual(scale, 1.0)
        self.assertEqual(ccy, "EUR")

    def test_lyif_alternative_ticker_eur(self):
        """LYI.F (alternative for LYI.DE) is EUR — currency from yfinance."""
        with self._patch_ticker("EUR", 22.50):
            scale, ccy = _detect_unit_scale("LYI.F")
        self.assertEqual(scale, 1.0)
        self.assertEqual(ccy, "EUR")

    def test_non_lse_empty_currency_falls_back_to_suffix_map(self):
        """yfinance returns empty currency for SAP.DE → fall back to EUR suffix hint."""
        with self._patch_ticker("", 0.0):
            scale, ccy = _detect_unit_scale("SAP.DE")
        self.assertEqual(scale, 1.0)
        self.assertEqual(ccy, "EUR")

    def test_non_lse_exception_falls_back_to_suffix_map(self):
        """yfinance raises for NESN.SW → fall back to CHF from suffix map."""
        with patch.object(resolver.yf, "Ticker", side_effect=RuntimeError("timeout")):
            scale, ccy = _detect_unit_scale("NESN.SW")
        self.assertEqual(scale, 1.0)
        self.assertEqual(ccy, "CHF")

    def test_unknown_suffix_falls_back_to_usd(self):
        """Ticker with unrecognised suffix → yfinance empty → default USD."""
        with self._patch_ticker("", 0.0):
            scale, ccy = _detect_unit_scale("FAKE.ZZ")
        self.assertEqual(scale, 1.0)
        self.assertEqual(ccy, "USD")

    # --- Caching ---

    def test_result_is_cached(self):
        """Second call for same symbol uses cache — yfinance called only once."""
        with self._patch_ticker("USD", 47.80) as mock_ticker:
            _detect_unit_scale("COPX.L")
            _detect_unit_scale("COPX.L")
        self.assertEqual(mock_ticker.call_count, 1)

    def test_cache_cleared_between_tests(self):
        """Cache is cleared in setUp so tests are independent."""
        with self._patch_ticker("USD", 47.80):
            scale1, ccy1 = _detect_unit_scale("COPX.L")
        _clear_caches_for_tests()
        with self._patch_ticker("GBX", 5050.0):
            scale2, ccy2 = _detect_unit_scale("COPX.L")
        # After cache clear the second patch takes effect
        self.assertEqual(ccy2, "GBP")
        self.assertEqual(scale2, 0.01)


class TestResolveIntegration(unittest.TestCase):
    """resolve() must propagate the correct currency into ResolvedTicker."""

    def setUp(self):
        _clear_caches_for_tests()

    def tearDown(self):
        _clear_caches_for_tests()

    def _resolve_with(self, symbol: str, yf_currency: str, last_price: float):
        ticker_mock = _mock_ticker(yf_currency, last_price)
        with patch.object(resolver.yf, "Ticker", return_value=ticker_mock):
            with patch.object(resolver, "_probe_sync", return_value=True):
                return resolver.resolve(symbol)

    def test_copx_l_resolve_gives_usd(self):
        """resolve('COPX.L') must return currency='USD', not 'GBP'."""
        rt = self._resolve_with("COPX.L", "USD", 47.80)
        self.assertEqual(rt.currency, "USD")
        self.assertEqual(rt.unit_scale, 1.0)

    def test_lloy_l_resolve_gives_gbp_scaled(self):
        """resolve('LLOY.L') with GBX → currency='GBP', unit_scale=0.01."""
        rt = self._resolve_with("LLOY.L", "GBX", 5050.0)
        self.assertEqual(rt.currency, "GBP")
        self.assertEqual(rt.unit_scale, 0.01)

    def test_sap_de_resolve_gives_eur(self):
        """resolve('SAP.DE') must return currency='EUR'."""
        rt = self._resolve_with("SAP.DE", "EUR", 210.0)
        self.assertEqual(rt.currency, "EUR")
        self.assertEqual(rt.unit_scale, 1.0)

    def test_no_suffix_us_stock_skips_probe(self):
        """US tickers (no suffix) skip yfinance currency probe; default USD."""
        with patch.object(resolver, "_probe_sync", return_value=True):
            with patch.object(resolver.yf, "Ticker") as mock_t:
                rt = resolver.resolve("AAPL")
        # _detect_unit_scale should NOT be called for bare US symbols
        self.assertEqual(rt.currency, "USD")
        self.assertEqual(rt.unit_scale, 1.0)
        mock_t.assert_not_called()


class TestPositionValueGBX(unittest.TestCase):
    """T212 currencyCode='GBX' in helpers.position_value() divides total by 100."""

    def test_t212_gbx_currency_code(self):
        from helpers import position_value
        pos = {
            "ticker": "LLOY_L_EQ",
            "quantity": 1000,
            "currentPrice": 5050,   # pence
            "averagePrice": 5050,   # same → pure_diff = 0, triggers currency fallback
            "ppl": 0,
            "fxPpl": 0,
            "currencyCode": "GBX",
        }
        val = position_value(pos)
        # 1000 shares × 5050p / 100 = £50,500
        self.assertAlmostEqual(val, 50_500.0, places=0)


if __name__ == "__main__":
    unittest.main()
