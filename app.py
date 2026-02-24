#!/usr/bin/env python3
"""
Streamlit v3.4 Options Flow Scanner (Replay + Contract IV + Underlying ATM IV 7D) — FIXED
======================================================================================
This version fixes your "IV always 0" issue by fixing the ROOT CAUSE:

1) SPOT was 0 for every row -> underlying ATM IV cannot be computed.
   - We now fetch spot with FALLBACKS:
        Polygon intraday -> Polygon prev close -> EODHD real-time -> EODHD last close
   - Spot will almost never be 0 anymore (unless ticker truly invalid).

2) Contract IV was 0 due to fragile parsing of EODHD chain field names.
   - We now read IV from multiple possible keys:
        impliedVolatility, implied_volatility, iv, IV

Outputs:
- Contract IV (EODHD chain exact contract if found)
- Underlying ATM IV (EODHD chain, nearest expiry within DTE window, strike closest to spot)
- Local IV store builds 7D history + ramp detection.

Secrets:
UW_TOKEN, POLYGON_API_KEY, EODHD_API_KEY

requirements.txt:
streamlit
requests
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

# -------------------- Constants --------------------

CT_OFFSET = -6
MAX_PENDING_TRADES = 50

PENDING_TRADES_FILE = "pending_trades.json"
INVERSE_SIGNALS_FILE = "inverse_signals.json"
VALIDATED_TRADES_FILE = "validated_trades.json"

SNAPSHOT_FILE = "last_uw_flows.json"
IV_STORE_FILE = "iv_history_store.json"

EXCLUDED_TICKERS_DEFAULT = {
    "SPX", "SPXW", "NDX", "VIX", "RUT", "DJX", "XSP", "OEX",
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC",
}


# -------------------- Helpers --------------------

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Tuple[int, Optional[Any], str]:
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        if resp.status_code == 200:
            try:
                return resp.status_code, resp.json(), ""
            except Exception:
                return resp.status_code, None, "Failed to parse JSON."
        return resp.status_code, None, f"HTTP {resp.status_code}: {resp.text[:400]}"
    except Exception as e:
        return 0, None, f"Request error: {e}"


def to_central_time(iso_timestamp: str, ct_offset_hours: int = CT_OFFSET) -> str:
    if not iso_timestamp:
        return ""
    try:
        if "T" in iso_timestamp:
            dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(iso_timestamp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        ct = dt + timedelta(hours=ct_offset_hours)
        return ct.strftime("%Y-%m-%d %I:%M:%S %p CT")
    except Exception:
        return iso_timestamp


def parse_uw_time_to_utc(iso_timestamp: str) -> Optional[datetime]:
    if not iso_timestamp:
        return None
    try:
        if "T" in iso_timestamp:
            dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        dt = datetime.strptime(iso_timestamp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def calculate_dte(expiry_yyyy_mm_dd: str) -> int:
    try:
        exp_date = datetime.strptime(expiry_yyyy_mm_dd, "%Y-%m-%d")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return (exp_date - today).days
    except Exception:
        return 0


def pretty_money(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)


def ensure_json_file(path: str, default_value: Any) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_value, f, indent=2)


def read_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json_file(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def today_yyyy_mm_dd() -> str:
    return date.today().strftime("%Y-%m-%d")


def contract_key(ticker: str, expiry: str, option_type: str, strike: float) -> str:
    return f"{ticker.upper()}|{expiry}|{option_type.lower()}|{float(strike):.2f}"


def underlying_iv_key(ticker: str) -> str:
    return f"{ticker.upper()}|UNDERLYING_ATM_IV"


# -------------------- Snapshot Manager --------------------

class SnapshotManager:
    def __init__(self, path: str = SNAPSHOT_FILE):
        self.path = path

    def save(self, flows: List[Dict[str, Any]]) -> None:
        payload = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "count": len(flows),
            "flows": flows,
        }
        write_json_file(self.path, payload)

    def load(self) -> List[Dict[str, Any]]:
        payload = read_json_file(self.path, {})
        if isinstance(payload, dict) and isinstance(payload.get("flows"), list):
            return payload["flows"]
        if isinstance(payload, list):
            return payload
        return []

    def get_meta(self) -> Dict[str, Any]:
        payload = read_json_file(self.path, {})
        if isinstance(payload, dict):
            return {"saved_at_utc": payload.get("saved_at_utc"), "count": payload.get("count")}
        return {}

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def raw_text(self) -> str:
        if not self.exists():
            return ""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""


# -------------------- Local IV Store --------------------

class LocalIVStore:
    def __init__(self, path: str = IV_STORE_FILE):
        self.path = path
        ensure_json_file(self.path, default_value={})

    def load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        data = read_json_file(self.path, {})
        return data if isinstance(data, dict) else {}

    def save_all(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        write_json_file(self.path, data)

    def upsert_today(self, key: str, iv_value: float) -> None:
        if iv_value <= 0:
            return
        data = self.load_all()
        rows = data.get(key, [])
        if not isinstance(rows, list):
            rows = []
        t = today_yyyy_mm_dd()

        replaced = False
        for r in rows:
            if isinstance(r, dict) and r.get("date") == t:
                r["iv"] = float(iv_value)
                replaced = True
                break
        if not replaced:
            rows.append({"date": t, "iv": float(iv_value)})

        rows = [r for r in rows if isinstance(r, dict) and r.get("date") and r.get("iv") is not None]
        rows.sort(key=lambda r: r["date"])
        rows = rows[-120:]

        data[key] = rows
        self.save_all(data)

    def get_history(self, key: str) -> Dict[str, float]:
        data = self.load_all()
        rows = data.get(key, [])
        out: Dict[str, float] = {}
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    continue
                d = str(r.get("date", ""))
                iv = safe_float(r.get("iv", 0))
                if d and iv > 0:
                    out[d] = iv
        return out

    def last_n_points(self, key: str, n: int = 7) -> List[Tuple[str, float]]:
        hist = self.get_history(key)
        if not hist:
            return []
        dates_sorted = sorted(hist.keys())
        last_dates = dates_sorted[-n:]
        return [(d, float(hist[d])) for d in last_dates]

    def detect_ramp(self, key: str, lookback_days: int = 3, require_strict: bool = True) -> Tuple[bool, List[Tuple[str, float]]]:
        pts = self.last_n_points(key, lookback_days)
        if len(pts) < lookback_days:
            return False, []
        vals = [v for _, v in pts]
        if require_strict:
            ok = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
        else:
            ok = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        return ok, pts

    def raw_text(self) -> str:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def reset(self) -> None:
        self.save_all({})


# -------------------- APIs --------------------

class UnusualWhalesAPI:
    BASE_URL = "https://api.unusualwhales.com/api"

    def __init__(self, token: str):
        self.token = token.strip()
        self.headers = {
            "Accept": "application/json, text/plain",
            "Authorization": f"Bearer {self.token}" if self.token else "",
        }

    def test_connection(self) -> Tuple[bool, str]:
        status, data, err = http_get(
            f"{self.BASE_URL}/option-trades/flow-alerts",
            headers=self.headers,
            params={"limit": 3},
        )
        if status == 200 and data:
            return True, "OK (UW flow alerts)"
        return False, err or "Failed"

    def get_flows(self, limit: int = 100) -> List[Dict[str, Any]]:
        status, data, _ = http_get(
            f"{self.BASE_URL}/option-trades/flow-alerts",
            headers=self.headers,
            params={"limit": limit},
        )
        if status != 200 or not data:
            return []
        return data.get("data", []) or []

    def get_ticker_flow(self, ticker: str, limit: int = 180) -> List[Dict[str, Any]]:
        ticker = ticker.strip().upper()
        status, data, _ = http_get(
            f"{self.BASE_URL}/stock/{ticker}/options-flow",
            headers=self.headers,
            params={"limit": limit},
        )
        if status == 200 and data:
            return data.get("data", []) or []
        return []

    def get_earnings(self, ticker: str) -> List[Dict[str, Any]]:
        ticker = ticker.strip().upper()
        status, data, _ = http_get(
            f"{self.BASE_URL}/stock/{ticker}/earnings-history",
            headers=self.headers,
        )
        if status == 200 and data:
            return data.get("data", []) or []
        return []


class PolygonAPI:
    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    def test_connection(self) -> Tuple[bool, str]:
        status, data, err = http_get(
            f"{self.BASE_URL}/v2/aggs/ticker/AAPL/prev",
            params={"apiKey": self.api_key},
        )
        if status == 200 and data and data.get("status") == "OK":
            return True, "OK (Polygon)"
        return False, err or "Failed"

    def get_previous_close(self, ticker: str) -> Tuple[float, str]:
        ticker = ticker.strip().upper()
        if not self.api_key:
            return 0.0, "Polygon: missing api key"
        status, data, err = http_get(
            f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/prev",
            params={"apiKey": self.api_key},
        )
        if status == 200 and data:
            results = data.get("results") or []
            if results:
                return safe_float(results[0].get("c", 0.0)), "Polygon prev close"
        return 0.0, f"Polygon prev close error: {err or 'no data'}"

    def get_spot_intraday(self, ticker: str, timestamp_ct: str) -> Tuple[float, str]:
        ticker = ticker.strip().upper()
        if not self.api_key:
            return 0.0, "Polygon: missing api key"
        try:
            if not timestamp_ct or "CT" not in timestamp_ct:
                return 0.0, "Polygon intraday: missing timestamp"
            parts = timestamp_ct.replace(" CT", "").strip()
            dt_ct = datetime.strptime(parts, "%Y-%m-%d %I:%M:%S %p")
            trade_date = dt_ct.strftime("%Y-%m-%d")

            url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{trade_date}/{trade_date}"
            status, data, err = http_get(url, params={"apiKey": self.api_key, "limit": 500})

            if status == 200 and data and data.get("results"):
                bars = data["results"]
                target_min = dt_ct.hour * 60 + dt_ct.minute

                closest = None
                best = 10_000
                for bar in bars:
                    bar_ts = safe_float(bar.get("t", 0)) / 1000.0
                    bar_dt_utc = datetime.utcfromtimestamp(bar_ts).replace(tzinfo=timezone.utc)
                    bar_ct = bar_dt_utc + timedelta(hours=CT_OFFSET)
                    bar_min = bar_ct.hour * 60 + bar_ct.minute
                    diff = abs(bar_min - target_min)
                    if diff < best:
                        best = diff
                        closest = bar

                if closest and best <= 5:
                    return safe_float(closest.get("c", 0.0)), "Polygon intraday"
            return 0.0, f"Polygon intraday error: {err or 'no bars'}"
        except Exception as e:
            return 0.0, f"Polygon intraday exception: {e}"

    def get_price_history(self, ticker: str, days: int = 30) -> List[Dict[str, Any]]:
        ticker = ticker.strip().upper()
        if not self.api_key:
            return []
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            url = (
                f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/"
                f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            )
            status, data, _ = http_get(url, params={"apiKey": self.api_key, "limit": days})
            if status == 200 and data:
                out: List[Dict[str, Any]] = []
                for r in (data.get("results") or []):
                    out.append(
                        {
                            "date": datetime.fromtimestamp(safe_float(r.get("t", 0)) / 1000.0).strftime("%Y-%m-%d"),
                            "open": safe_float(r.get("o", 0)),
                            "high": safe_float(r.get("h", 0)),
                            "low": safe_float(r.get("l", 0)),
                            "close": safe_float(r.get("c", 0)),
                            "volume": safe_float(r.get("v", 0)),
                        }
                    )
                return out
            return []
        except Exception:
            return []

    def calculate_support_resistance(self, ticker: str, strike: float) -> Dict[str, Any]:
        candles = self.get_price_history(ticker, 30)
        if not candles or len(candles) < 5:
            return {}
        highs = [safe_float(c["high"]) for c in candles]
        lows = [safe_float(c["low"]) for c in candles]

        resistance = max(highs[-10:]) if len(highs) >= 10 else max(highs)
        support = min(lows[-10:]) if len(lows) >= 10 else min(lows)

        recent_high_wick = max(highs[-5:]) if len(highs) >= 5 else max(highs)
        recent_low_wick = min(lows[-5:]) if len(lows) >= 5 else min(lows)

        wick_triggered = (abs(strike - recent_high_wick) < 0.5) or (abs(strike - recent_low_wick) < 0.5)
        return {
            "resistance": resistance,
            "support": support,
            "recent_high_wick": recent_high_wick,
            "recent_low_wick": recent_low_wick,
            "wick_triggered": wick_triggered,
        }


class EODHDAPI:
    BASE_URL = "https://eodhd.com/api"

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self._chain_cache: Dict[str, Dict[str, Any]] = {}

    def test_connection(self) -> Tuple[bool, str]:
        status, data, err = http_get(
            f"{self.BASE_URL}/eod/AAPL.US",
            params={"api_token": self.api_key, "fmt": "json", "from": "2025-01-01", "limit": 1},
        )
        if status == 200 and data:
            return True, "OK (EODHD)"
        return False, err or "Failed"

    def get_spot_realtime(self, ticker: str) -> Tuple[float, str]:
        """
        Try EODHD real-time spot. If market closed, may still return last/close-ish.
        Endpoint: /real-time/{ticker}.US
        """
        ticker = ticker.strip().upper()
        if not self.api_key:
            return 0.0, "EODHD spot: missing api key"
        url = f"{self.BASE_URL}/real-time/{ticker}.US"
        status, data, err = http_get(url, params={"api_token": self.api_key, "fmt": "json"}, timeout=20)
        if status == 200 and isinstance(data, dict):
            # EODHD real-time responses vary; try common keys
            for k in ("close", "price", "last", "last_price"):
                v = safe_float(data.get(k, 0))
                if v > 0:
                    return v, f"EODHD real-time ({k})"
        return 0.0, f"EODHD real-time error: {err or 'no price keys'}"

    def get_spot_last_close(self, ticker: str) -> Tuple[float, str]:
        ticker = ticker.strip().upper()
        if not self.api_key:
            return 0.0, "EODHD last close: missing api key"
        url = f"{self.BASE_URL}/eod/{ticker}.US"
        status, data, err = http_get(
            url,
            params={"api_token": self.api_key, "fmt": "json", "from": "2020-01-01", "limit": 1},
            timeout=20,
        )
        if status == 200 and isinstance(data, list) and data:
            c = safe_float(data[0].get("close", 0))
            if c > 0:
                return c, "EODHD last close"
        return 0.0, f"EODHD last close error: {err or 'no data'}"

    def get_options_chain(self, ticker: str) -> Tuple[Optional[Dict[str, Any]], str]:
        ticker = ticker.strip().upper()
        if not self.api_key:
            return None, "EODHD: missing api key"
        if ticker in self._chain_cache:
            return self._chain_cache[ticker], "EODHD chain (cached)"
        url = f"{self.BASE_URL}/options/{ticker}.US"
        status, data, err = http_get(url, params={"api_token": self.api_key, "fmt": "json"}, timeout=30)
        if status != 200 or not data or not isinstance(data, dict):
            return None, f"EODHD chain error: {err or 'no data'}"
        self._chain_cache[ticker] = data
        return data, "EODHD chain (fresh)"

    def _normalize_iv_pct(self, iv: float) -> float:
        if iv <= 0:
            return 0.0
        return iv * 100.0 if iv < 1 else iv

    def _extract_iv_from_option_row(self, opt: Dict[str, Any]) -> float:
        """
        EODHD can use different keys depending on feed/version.
        Try multiple:
        - impliedVolatility
        - implied_volatility
        - iv
        - IV
        """
        if not isinstance(opt, dict):
            return 0.0
        for k in ("impliedVolatility", "implied_volatility", "iv", "IV"):
            iv = self._normalize_iv_pct(safe_float(opt.get(k, 0)))
            if iv > 0:
                return float(iv)
        return 0.0

    def get_iv_from_chain(self, ticker: str, strike: float, expiry: str, option_type: str) -> Tuple[float, str]:
        """
        Return (IV%, reason).
        """
        ticker = ticker.strip().upper()
        option_type = option_type.lower().strip()

        chain, chain_src = self.get_options_chain(ticker)
        if not chain:
            return 0.0, "No chain"

        best_match_iv = 0.0
        best_match_diff = 1e9
        saw_expiry = False

        for exp_key, exp_payload in chain.items():
            if expiry not in str(exp_key):
                continue
            saw_expiry = True
            if not isinstance(exp_payload, dict):
                continue

            options_list = exp_payload.get(option_type + "s", [])
            if isinstance(options_list, dict):
                options_list = list(options_list.values())
            if not isinstance(options_list, list):
                continue

            for opt in options_list:
                if not isinstance(opt, dict):
                    continue
                opt_strike = safe_float(opt.get("strike", 0))
                diff = abs(opt_strike - strike)
                if diff < best_match_diff:
                    best_match_diff = diff
                    best_match_iv = self._extract_iv_from_option_row(opt)

        if not saw_expiry:
            return 0.0, f"{chain_src} | expiry not found"
        if best_match_diff > 0.51:
            return 0.0, f"{chain_src} | strike not close (best diff {best_match_diff:.2f})"
        if best_match_iv <= 0:
            return 0.0, f"{chain_src} | IV field missing"
        return float(best_match_iv), f"{chain_src} | ok (diff {best_match_diff:.2f})"

    def get_underlying_atm_iv(
        self,
        ticker: str,
        spot: float,
        dte_min: int = 7,
        dte_max: int = 30,
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Underlying ATM IV from EODHD chain:
        - Find expiries in DTE window
        - Pick nearest expiry
        - Pick strike closest to spot
        - Use CALL IV, fallback PUT IV
        """
        dbg: Dict[str, Any] = {"ticker": ticker, "spot": spot, "dte_min": dte_min, "dte_max": dte_max}

        if spot <= 0:
            return 0.0, "Underlying IV: spot is 0", dbg

        chain, chain_src = self.get_options_chain(ticker)
        dbg["chain_source"] = chain_src
        if not chain:
            return 0.0, "Underlying IV: no chain", dbg

        expiries: List[str] = []
        for exp_key in chain.keys():
            exp = str(exp_key)[:10]
            if len(exp) == 10 and exp[4] == "-" and exp[7] == "-":
                dte = calculate_dte(exp)
                if dte_min <= dte <= dte_max:
                    expiries.append(exp)

        expiries = sorted(set(expiries), key=lambda e: calculate_dte(e))
        dbg["eligible_expiries"] = expiries[:30]
        if not expiries:
            return 0.0, f"Underlying IV: no expiries in {dte_min}-{dte_max} DTE", dbg

        expiry = expiries[0]
        dbg["chosen_expiry"] = expiry

        exp_payload = None
        for exp_key, payload in chain.items():
            if expiry in str(exp_key) and isinstance(payload, dict):
                exp_payload = payload
                break
        if not exp_payload:
            return 0.0, "Underlying IV: expiry payload missing", dbg

        calls = exp_payload.get("calls", [])
        puts = exp_payload.get("puts", [])
        if isinstance(calls, dict):
            calls = list(calls.values())
        if isinstance(puts, dict):
            puts = list(puts.values())
        calls = calls if isinstance(calls, list) else []
        puts = puts if isinstance(puts, list) else []

        strikes: List[float] = []
        for o in calls:
            s = safe_float(o.get("strike", 0)) if isinstance(o, dict) else 0
            if s > 0:
                strikes.append(s)
        if not strikes:
            for o in puts:
                s = safe_float(o.get("strike", 0)) if isinstance(o, dict) else 0
                if s > 0:
                    strikes.append(s)

        strikes = sorted(set(strikes))
        if not strikes:
            return 0.0, "Underlying IV: no strikes", dbg

        atm = min(strikes, key=lambda s: abs(s - spot))
        dbg["atm_strike"] = atm

        def nearest_iv(opts: List[Dict[str, Any]]) -> float:
            best_iv = 0.0
            best_diff = 1e9
            for opt in opts:
                if not isinstance(opt, dict):
                    continue
                s = safe_float(opt.get("strike", 0))
                diff = abs(s - atm)
                if diff < best_diff:
                    best_diff = diff
                    best_iv = self._extract_iv_from_option_row(opt)
            return float(best_iv) if best_diff <= 0.51 else 0.0

        iv_call = nearest_iv(calls)
        iv_put = nearest_iv(puts)
        dbg["iv_call"] = iv_call
        dbg["iv_put"] = iv_put

        if iv_call > 0:
            return iv_call, f"{chain_src} | ATM IV CALL {expiry} {atm}", dbg
        if iv_put > 0:
            return iv_put, f"{chain_src} | ATM IV PUT {expiry} {atm}", dbg

        return 0.0, f"{chain_src} | ATM IV missing fields", dbg


# -------------------- Ladder --------------------

class LadderDetector:
    def __init__(self, uw: UnusualWhalesAPI):
        self.uw = uw

    def detect(
        self,
        ticker: str,
        target_strike: float,
        option_type: str,
        expiry: str,
        recent_minutes: int = 90,
        min_unique_strikes: int = 3,
    ) -> Tuple[bool, List[float]]:
        if not self.uw.token:
            return False, []
        flows = self.uw.get_ticker_flow(ticker, limit=180)
        if not flows:
            return False, []

        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(minutes=recent_minutes)

        strikes: set[float] = {float(target_strike)}
        for f in flows:
            f_type = str(f.get("option_type", "")).lower().strip()
            f_expiry = str(f.get("expiry", "")).strip()
            if f_type != option_type or f_expiry != expiry:
                continue
            ts = parse_uw_time_to_utc(str(f.get("start_time", "")))
            if ts is None or ts < cutoff:
                continue
            f_strike = safe_float(f.get("strike", 0))
            if f_strike > 0:
                strikes.add(float(f_strike))

        if len(strikes) >= min_unique_strikes:
            related = sorted([s for s in strikes if abs(s - target_strike) > 1e-9])
            return True, related
        return False, []


# -------------------- Enrichment --------------------

class DataEnricher:
    def __init__(
        self,
        uw: UnusualWhalesAPI,
        polygon: PolygonAPI,
        eodhd: EODHDAPI,
        iv_store: LocalIVStore,
        iv_lookback_days: int,
        iv_require_strict: bool,
        underlying_iv_dte_min: int,
        underlying_iv_dte_max: int,
    ):
        self.uw = uw
        self.polygon = polygon
        self.eodhd = eodhd
        self.iv_store = iv_store
        self.iv_lookback_days = iv_lookback_days
        self.iv_require_strict = iv_require_strict
        self.underlying_iv_dte_min = underlying_iv_dte_min
        self.underlying_iv_dte_max = underlying_iv_dte_max

    def get_best_spot(self, ticker: str, timestamp_ct: str) -> Tuple[float, str]:
        """
        FIX: If Polygon fails, spot becomes 0 -> IV breaks.
        So we do a strict fallback ladder:
        1) Polygon intraday (if timestamp)
        2) Polygon prev close
        3) EODHD real-time
        4) EODHD last close
        """
        # 1) Polygon intraday
        p1, s1 = self.polygon.get_spot_intraday(ticker, timestamp_ct)
        if p1 > 0:
            return p1, s1

        # 2) Polygon prev close
        p2, s2 = self.polygon.get_previous_close(ticker)
        if p2 > 0:
            return p2, s2

        # 3) EODHD real-time
        p3, s3 = self.eodhd.get_spot_realtime(ticker)
        if p3 > 0:
            return p3, s3

        # 4) EODHD last close
        p4, s4 = self.eodhd.get_spot_last_close(ticker)
        if p4 > 0:
            return p4, s4

        return 0.0, f"Spot failed: {s1} | {s2} | {s3} | {s4}"

    def enrich_trade(self, raw_flow: Dict[str, Any], use_contract_iv: bool, use_underlying_iv: bool) -> Dict[str, Any]:
        ticker = str(raw_flow.get("ticker", "")).upper().strip()
        strike = safe_float(raw_flow.get("strike", 0))
        option_type = str(raw_flow.get("option_type", "call")).lower().strip()
        expiry = str(raw_flow.get("expiry", "")).strip()
        timestamp = str(raw_flow.get("start_time", "")).strip()

        timestamp_ct = to_central_time(timestamp)

        # FIX: robust spot
        spot, spot_source = self.get_best_spot(ticker, timestamp_ct)

        # S/R only if polygon has history access
        sr_data = self.polygon.calculate_support_resistance(ticker, strike) if self.polygon.api_key else {}

        # Premium + Ask%
        total_prem = (
            safe_float(raw_flow.get("total_ask_side_prem", 0))
            + safe_float(raw_flow.get("total_bid_side_prem", 0))
            + safe_float(raw_flow.get("total_mid_side_prem", 0))
            + safe_float(raw_flow.get("total_no_side_prem", 0))
        )
        ask_prem = safe_float(raw_flow.get("total_ask_side_prem", 0))
        ask_pct = (ask_prem / total_prem * 100.0) if total_prem > 0 else 0.0

        volume = safe_int(raw_flow.get("total_size", 0))
        oi = safe_int(raw_flow.get("open_interest", 0))
        vol_oi_ratio = (volume / oi) if oi > 0 else 999.0

        # Premium % (only valid if spot>0)
        denom = (spot * 100.0 * max(volume, 1)) if spot > 0 else 0.0
        premium_pct = (total_prem / denom * 100.0) if denom > 0 else 0.0

        # Strike distance
        if spot > 0:
            strike_dist_pct = abs(strike - spot) / spot * 100.0
        else:
            strike_dist_pct = 0.0

        dte = calculate_dte(expiry)

        # Earnings
        days_to_er: Optional[int] = None
        if self.uw.token:
            earnings = self.uw.get_earnings(ticker)
            if earnings:
                today_dt = datetime.now()
                for er in earnings:
                    er_date_str = str(er.get("date", "")).strip()
                    try:
                        er_date = datetime.strptime(er_date_str, "%Y-%m-%d")
                        delta = (er_date - today_dt).days
                        if 0 <= delta <= 30:
                            days_to_er = delta
                            break
                    except Exception:
                        continue

        # Keys
        ckey = contract_key(ticker, expiry, option_type, strike)
        ukey = underlying_iv_key(ticker)

        # Contract IV
        current_iv_contract = 0.0
        contract_iv_reason = ""
        if use_contract_iv and self.eodhd.api_key:
            current_iv_contract, contract_iv_reason = self.eodhd.get_iv_from_chain(ticker, strike, expiry, option_type)
            if current_iv_contract > 0:
                self.iv_store.upsert_today(ckey, current_iv_contract)

        contract_iv_history = self.iv_store.get_history(ckey)
        contract_iv_ramping, contract_ramp_points = self.iv_store.detect_ramp(
            ckey, lookback_days=self.iv_lookback_days, require_strict=self.iv_require_strict
        )

        # Underlying IV
        underlying_iv_current = 0.0
        underlying_iv_source = ""
        underlying_iv_debug: Dict[str, Any] = {}
        if use_underlying_iv and self.eodhd.api_key:
            underlying_iv_current, underlying_iv_source, underlying_iv_debug = self.eodhd.get_underlying_atm_iv(
                ticker=ticker,
                spot=spot,
                dte_min=self.underlying_iv_dte_min,
                dte_max=self.underlying_iv_dte_max,
            )
            if underlying_iv_current > 0:
                self.iv_store.upsert_today(ukey, underlying_iv_current)

        underlying_points_7d = self.iv_store.last_n_points(ukey, 7)
        iv7d_change = 0.0
        if len(underlying_points_7d) >= 2:
            iv7d_change = float(underlying_points_7d[-1][1] - underlying_points_7d[0][1])

        underlying_ramping, underlying_ramp_pts = self.iv_store.detect_ramp(
            ukey, lookback_days=self.iv_lookback_days, require_strict=self.iv_require_strict
        )

        clean_exception = (
            ask_pct >= 70 and vol_oi_ratio > 1 and strike_dist_pct <= 7 and 2.5 <= premium_pct <= 5.0
        )

        return {
            "ticker": ticker,
            "strike": strike,
            "option_type": option_type,
            "expiry": expiry,
            "entry_timestamp": timestamp_ct,

            "spot": float(spot),
            "spot_source": spot_source,

            "strike_dist_pct": float(strike_dist_pct),
            "total_premium": float(total_prem),
            "premium_pct": float(premium_pct),
            "volume": int(volume),
            "open_interest": int(oi),
            "vol_oi_ratio": float(vol_oi_ratio),
            "ask_pct": float(ask_pct),
            "dte": int(dte),

            "wick_triggered": bool(sr_data.get("wick_triggered", False)),
            "support": safe_float(sr_data.get("support", 0)),
            "resistance": safe_float(sr_data.get("resistance", 0)),

            "contract_key": ckey,
            "current_iv": float(current_iv_contract),
            "contract_iv_reason": contract_iv_reason,
            "iv_history_local": contract_iv_history,
            "iv_ramping": bool(contract_iv_ramping),
            "iv_ramp_points": contract_ramp_points,

            "underlying_iv_key": ukey,
            "iv_underlying_current": float(underlying_iv_current),
            "iv_underlying_source": underlying_iv_source,
            "iv_underlying_points_7d": underlying_points_7d,
            "iv_underlying_7d_change": float(iv7d_change),
            "iv_underlying_ramping": bool(underlying_ramping),
            "iv_underlying_ramp_points": underlying_ramp_pts,
            "iv_underlying_debug": underlying_iv_debug,

            "iv_ramp_lookback_days": int(self.iv_lookback_days),
            "days_to_earnings": days_to_er,
            "clean_exception": bool(clean_exception),

            "has_sweep": bool(raw_flow.get("is_sweep", False)),
            "ladder_role": "isolated",
            "related_strikes": [],
            "category_tags": [],
            "_raw": raw_flow,
        }


# -------------------- Scoring --------------------

class V34ScoringEngine:
    def score(self, record: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        factors: List[str] = []
        penalties: List[str] = []
        tags: List[str] = []

        # Spot sanity
        if safe_float(record.get("spot", 0)) <= 0:
            score -= 3
            penalties.append("Spot=0 (data broken) (-3)")
            tags.append("BadSpot")

        prem_pct = safe_float(record.get("premium_pct", 0))
        if 2.5 <= prem_pct <= 5.0:
            score += 2
            factors.append(f"Premium {prem_pct:.1f}% (+2)")
        elif 1.0 <= prem_pct < 2.5:
            score += 1
            factors.append(f"Premium {prem_pct:.1f}% (+1)")
        elif prem_pct < 1.0 and safe_float(record.get("spot", 0)) > 0:
            score -= 2
            penalties.append(f"Ultra-low premium {prem_pct:.2f}% (-2)")

        dist = safe_float(record.get("strike_dist_pct", 0))
        if dist <= 7 and safe_float(record.get("spot", 0)) > 0:
            score += 2
            factors.append(f"Strike {dist:.1f}% OTM (+2)")
        elif dist > 15 and safe_float(record.get("spot", 0)) > 0:
            score -= 2
            penalties.append(f"Strike {dist:.1f}% deep OTM (-2)")

        dte = safe_int(record.get("dte", 0))
        if 7 <= dte <= 21:
            score += 1
            factors.append(f"DTE {dte}d (+1)")
        elif dte <= 1:
            score -= 1
            penalties.append("0-1 DTE (-1)")

        ask_pct = safe_float(record.get("ask_pct", 0))
        if ask_pct >= 70:
            score += 1
            factors.append(f"Ask {ask_pct:.0f}% (+1)")
        elif 0 < ask_pct < 30:
            score -= 2
            penalties.append(f"Bid/mid heavy (Ask {ask_pct:.0f}%) (-2)")

        vol_oi = safe_float(record.get("vol_oi_ratio", 0))
        if vol_oi >= 2:
            score += 2
            factors.append(f"Vol/OI {vol_oi:.1f}x (+2)")
        elif vol_oi >= 1:
            score += 1
            factors.append(f"Vol/OI {vol_oi:.1f}x (+1)")

        if bool(record.get("wick_triggered", False)):
            score -= 2
            penalties.append("Wick reversal strike (-2)")

        # IV signals (prefer underlying)
        if safe_float(record.get("iv_underlying_current", 0)) > 0:
            tags.append("HasUnderlyingIV")
            if bool(record.get("iv_underlying_ramping", False)):
                score += 1
                factors.append(f"Underlying IV ramp (+1) {record.get('iv_underlying_ramp_points')}")
            iv7 = safe_float(record.get("iv_underlying_7d_change", 0))
            if iv7 >= 3.0:
                score += 1
                factors.append(f"Underlying IV +{iv7:.1f} in 7d (+1)")
            elif iv7 <= -3.0:
                score -= 1
                penalties.append(f"Underlying IV {iv7:.1f} in 7d (-1)")
        else:
            tags.append("NoUnderlyingIV")

        if safe_float(record.get("current_iv", 0)) > 0:
            tags.append("HasContractIV")
            if bool(record.get("iv_ramping", False)):
                score += 1
                factors.append(f"Contract IV ramp (+1) {record.get('iv_ramp_points')}")
        else:
            tags.append("NoContractIV")

        ladder_role = str(record.get("ladder_role", "isolated")).lower()
        if ladder_role in ("anchor", "specleg", "ladder"):
            score += 1
            factors.append("Ladder/cluster (+1)")
        else:
            score -= 1
            penalties.append("Isolated (-1)")

        days_to_er = record.get("days_to_earnings")
        if isinstance(days_to_er, int) and 2 <= days_to_er <= 10:
            score += 1
            factors.append(f"Catalyst {days_to_er}d (+1)")
            tags.append("PreER")

        # Cap logic
        if safe_float(record.get("iv_underlying_current", 0)) > 0 or safe_float(record.get("current_iv", 0)) > 0:
            max_score = 12
        elif bool(record.get("clean_exception", False)):
            max_score = 7
        else:
            max_score = 6

        final_score = min(score, max_score)

        if final_score >= 8:
            verdict = "HIGH CONVICTION"
        elif final_score >= 7:
            verdict = "TRADEABLE"
        elif final_score >= 6:
            verdict = "MODERATE"
        elif final_score >= 5:
            verdict = "WATCHLIST"
        else:
            verdict = "TRAP / SKIP"

        record["predictive_score"] = int(final_score)
        record["max_score"] = int(max_score)
        record["score_factors"] = factors
        record["score_penalties"] = penalties
        record["verdict"] = verdict
        record["category_tags"] = list(dict.fromkeys(tags))
        return record


# -------------------- Queue --------------------

class QueueManager:
    def __init__(self, pending_file: str, inverse_file: str, validated_file: str):
        self.pending_file = pending_file
        self.inverse_file = inverse_file
        self.validated_file = validated_file
        ensure_json_file(self.pending_file, default_value=[])
        ensure_json_file(self.inverse_file, default_value=[])
        ensure_json_file(self.validated_file, default_value=[])

    def load_queue(self, filepath: str) -> List[Dict[str, Any]]:
        if not os.path.exists(filepath):
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                x = json.load(f)
                return x if isinstance(x, list) else []
        except Exception:
            return []

    def save_queue(self, filepath: str, data: List[Dict[str, Any]]) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def add_pending(self, trade: Dict[str, Any]) -> None:
        q = self.load_queue(self.pending_file)
        q.append(trade)
        if len(q) > MAX_PENDING_TRADES:
            q = q[-MAX_PENDING_TRADES:]
        self.save_queue(self.pending_file, q)

    def add_inverse(self, trade: Dict[str, Any]) -> None:
        q = self.load_queue(self.inverse_file)
        q.append(trade)
        self.save_queue(self.inverse_file, q)


# -------------------- UI --------------------

st.set_page_config(page_title="v3.4 Options Flow Scanner", layout="wide")
st.title("v3.4 Options Flow Scanner (FIXED Spot + EODHD IV)")
st.caption("If IV is 0, 99% of the time it was because Spot was 0. This version fixes that.")

snapshot = SnapshotManager(SNAPSHOT_FILE)
iv_store = LocalIVStore(IV_STORE_FILE)

with st.sidebar:
    st.header("Data Source")
    data_mode = st.toggle("Replay Mode (ON = use saved snapshot)", value=False)
    meta = snapshot.get_meta()
    if meta.get("saved_at_utc"):
        st.write(f"Snapshot saved: `{meta.get('saved_at_utc')}`")
        st.write(f"Snapshot count: `{meta.get('count')}`")
    else:
        st.write("Snapshot saved: —")

    st.divider()
    st.header("API Keys")
    uw_token = st.text_input("Unusual Whales Token", value=os.getenv("UW_TOKEN", ""), type="password")
    polygon_key = st.text_input("Polygon API Key", value=os.getenv("POLYGON_API_KEY", ""), type="password")
    eodhd_key = st.text_input("EODHD API Key", value=os.getenv("EODHD_API_KEY", ""), type="password")

    st.divider()
    st.subheader("IV Settings")
    iv_lookback_days = st.slider("Ramp lookback days", 3, 10, 3, 1)
    iv_strict = st.checkbox("Require strictly increasing IV", value=True)

    st.markdown("**Underlying ATM IV Window (EODHD)**")
    u_dte_min = st.slider("Underlying IV min DTE", 3, 30, 7, 1)
    u_dte_max = st.slider("Underlying IV max DTE", 7, 90, 30, 1)

    st.divider()
    st.subheader("Scan Controls")
    use_iv_contract = st.checkbox("Use contract IV (EODHD) + store", value=True)
    use_iv_underlying = st.checkbox("Use underlying ATM IV (EODHD) + store 7D", value=True)
    limit = st.slider("UW flow alerts limit (Live only)", 10, 250, 200, 10)

    min_premium = st.number_input("Min premium ($)", min_value=0, value=25_000, step=5_000)
    min_size = st.number_input("Min size (contracts)", min_value=0, value=0, step=100)
    min_vol_oi = st.number_input("Min Vol/OI", min_value=0.0, value=1.0, step=0.1)

    require_vol_gt_oi = st.checkbox("Require Vol > OI", value=False)
    exclude_indices = st.checkbox("Exclude indices + major ETFs", value=True)

    st.divider()
    st.subheader("Ladder Settings")
    ladder_minutes = st.slider("Ladder time window (minutes)", 15, 240, 90, 15)
    ladder_min_strikes = st.slider("Min unique strikes to call a ladder", 2, 6, 3, 1)

    st.divider()
    st.subheader("Local IV Store")
    st.download_button(
        "Download IV store",
        data=iv_store.raw_text(),
        file_name="iv_history_store.json",
        mime="application/json",
        use_container_width=True,
    )
    if st.button("Reset IV store (danger)", type="secondary", use_container_width=True):
        iv_store.reset()
        st.warning("IV store reset.")

# Instantiate
uw = UnusualWhalesAPI(uw_token)
polygon = PolygonAPI(polygon_key)
eodhd = EODHDAPI(eodhd_key)

enricher = DataEnricher(
    uw=uw,
    polygon=polygon,
    eodhd=eodhd,
    iv_store=iv_store,
    iv_lookback_days=int(iv_lookback_days),
    iv_require_strict=bool(iv_strict),
    underlying_iv_dte_min=int(u_dte_min),
    underlying_iv_dte_max=int(u_dte_max),
)
scorer = V34ScoringEngine()
ladder = LadderDetector(uw)
queue = QueueManager(PENDING_TRADES_FILE, INVERSE_SIGNALS_FILE, VALIDATED_TRADES_FILE)

tabs = st.tabs(["Scan", "Debug Spot/IV", "Connections"])


def get_source_flows() -> Tuple[List[Dict[str, Any]], str]:
    if data_mode:
        return snapshot.load(), "Replay (snapshot)"
    return uw.get_flows(limit=limit), "Live (UW API)"


# ---- Scan Tab ----
with tabs[0]:
    st.subheader("Run Scanner")

    colA, colB = st.columns([1, 2], gap="large")
    with colA:
        run = st.button("Run scan", type="primary", use_container_width=True)
        st.caption("If market is closed, turn ON Replay Mode.")
    with colB:
        st.markdown(
            """
- FIXED: Spot fallback chain so Spot will not be 0 (Polygon -> EODHD).
- FIXED: Contract IV parsing reads multiple IV field names.
- Underlying ATM IV now works because Spot works.
"""
        )

    if run:
        if not data_mode and not uw_token:
            st.error("Live Mode requires UW_TOKEN.")
        else:
            skip_reasons = {"premium": 0, "excluded": 0, "vol_oi": 0, "min_size": 0, "min_vol_oi": 0, "bad_ticker": 0}

            with st.spinner("Loading flows..."):
                flows, src = get_source_flows()

            if not flows:
                st.warning(f"No flows from {src}.")
            else:
                if not data_mode:
                    try:
                        snapshot.save(flows)
                    except Exception:
                        pass

                excluded = EXCLUDED_TICKERS_DEFAULT if exclude_indices else set()
                results: List[Dict[str, Any]] = []
                skipped = 0

                with st.spinner("Filtering, enriching, laddering, scoring..."):
                    for f in flows:
                        ticker = str(f.get("ticker", "")).upper().strip()
                        if not ticker or len(ticker) > 8:
                            skipped += 1
                            skip_reasons["bad_ticker"] += 1
                            continue

                        total_prem = (
                            safe_float(f.get("total_ask_side_prem", 0))
                            + safe_float(f.get("total_bid_side_prem", 0))
                            + safe_float(f.get("total_mid_side_prem", 0))
                            + safe_float(f.get("total_no_side_prem", 0))
                        )
                        if total_prem < float(min_premium):
                            skipped += 1
                            skip_reasons["premium"] += 1
                            continue

                        if ticker in excluded:
                            skipped += 1
                            skip_reasons["excluded"] += 1
                            continue

                        vol = safe_int(f.get("total_size", 0))
                        oi = safe_int(f.get("open_interest", 0))

                        if vol < int(min_size):
                            skipped += 1
                            skip_reasons["min_size"] += 1
                            continue

                        if require_vol_gt_oi and (oi > 0) and (vol <= oi):
                            skipped += 1
                            skip_reasons["vol_oi"] += 1
                            continue

                        vol_oi_ratio = (vol / oi) if oi > 0 else 999.0
                        if vol_oi_ratio < float(min_vol_oi):
                            skipped += 1
                            skip_reasons["min_vol_oi"] += 1
                            continue

                        enriched = enricher.enrich_trade(
                            f,
                            use_contract_iv=bool(use_iv_contract and eodhd_key),
                            use_underlying_iv=bool(use_iv_underlying and eodhd_key),
                        )

                        is_ladder, related = ladder.detect(
                            ticker=enriched["ticker"],
                            target_strike=enriched["strike"],
                            option_type=enriched["option_type"],
                            expiry=enriched["expiry"],
                            recent_minutes=int(ladder_minutes),
                            min_unique_strikes=int(ladder_min_strikes),
                        )
                        if is_ladder:
                            enriched["ladder_role"] = "ladder"
                            enriched["related_strikes"] = related

                        scored = scorer.score(enriched)
                        results.append(scored)

                        if safe_int(scored.get("predictive_score", 0)) >= 5:
                            queue.add_pending(scored)
                        if safe_int(scored.get("predictive_score", 0)) <= -3:
                            queue.add_inverse(scored)

                st.success(f"Source: {src} • Scored {len(results)} trades • Skipped {skipped}")
                st.write("Skip breakdown:", skip_reasons)

                if results:
                    results.sort(
                        key=lambda r: (safe_int(r.get("predictive_score", 0)), safe_float(r.get("total_premium", 0))),
                        reverse=True,
                    )

                    table = []
                    for r in results:
                        table.append(
                            {
                                "Ticker": r.get("ticker"),
                                "Type": str(r.get("option_type", "")).upper(),
                                "Strike": r.get("strike"),
                                "Expiry": r.get("expiry"),
                                "Spot": round(safe_float(r.get("spot", 0.0)), 2),
                                "SpotSrc": str(r.get("spot_source", ""))[:40],
                                "Premium$": round(safe_float(r.get("total_premium", 0.0))),
                                "Ask%": round(safe_float(r.get("ask_pct", 0.0)), 1),
                                "Vol/OI": round(safe_float(r.get("vol_oi_ratio", 0.0)), 2),
                                "IV(Contract)%": round(safe_float(r.get("current_iv", 0.0)), 2),
                                "IV(Underlying)%": round(safe_float(r.get("iv_underlying_current", 0.0)), 2),
                                "IV7dΔ": round(safe_float(r.get("iv_underlying_7d_change", 0.0)), 2),
                                "IVRamp(U)": bool(r.get("iv_underlying_ramping", False)),
                                "Score": r.get("predictive_score"),
                                "Verdict": r.get("verdict"),
                                "Tags": ", ".join(r.get("category_tags", [])),
                            }
                        )
                    st.dataframe(table, use_container_width=True, hide_index=True)

                    st.divider()
                    st.subheader("Details (Top 40)")
                    for r in results[:40]:
                        header = (
                            f"{r['ticker']} {str(r['option_type']).upper()} ${r['strike']} {r['expiry']} • "
                            f"Spot {safe_float(r.get('spot',0)):.2f} • IVu {safe_float(r.get('iv_underlying_current',0)):.2f}% • "
                            f"IVc {safe_float(r.get('current_iv',0)):.2f}% • Score {r['predictive_score']}"
                        )
                        with st.expander(header, expanded=False):
                            st.write("**Spot Source**:", r.get("spot_source"))
                            st.write("**Contract IV reason**:", r.get("contract_iv_reason"))
                            st.write("**Underlying IV source**:", r.get("iv_underlying_source"))
                            st.write("**Underlying IV debug**:")
                            st.json(r.get("iv_underlying_debug", {}))
                            st.write("**Underlying IV 7D points:**")
                            st.json(r.get("iv_underlying_points_7d", []))
                            st.write("**Raw UW (debug):**")
                            st.json(r.get("_raw", {}))


# ---- Debug Tab ----
with tabs[1]:
    st.subheader("Debug Spot/IV (this tells you EXACTLY why IV is 0)")
    tkr = st.text_input("Ticker", value="AAPL")
    exp = st.text_input("Expiry (YYYY-MM-DD)", value=(datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"))
    strike = st.number_input("Strike", min_value=0.0, value=200.0, step=1.0)
    opt_type = st.selectbox("Type", ["call", "put"], index=0)

    if st.button("Run Debug", type="primary"):
        ts = to_central_time(datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        spot, src = enricher.get_best_spot(tkr.upper(), ts)
        st.write("Spot:", spot, "| source:", src)

        ivc, ivc_reason = eodhd.get_iv_from_chain(tkr.upper(), float(strike), exp, opt_type)
        st.write("Contract IV:", ivc, "| reason:", ivc_reason)

        ivu, ivu_src, dbg = eodhd.get_underlying_atm_iv(tkr.upper(), spot, int(u_dte_min), int(u_dte_max))
        st.write("Underlying ATM IV:", ivu, "| source:", ivu_src)
        st.json(dbg)

        st.caption("If Spot is OK but IV is 0 and debug shows iv_call=0 & iv_put=0, EODHD chain does not include IV for that ticker/expiry.")


# ---- Connections ----
with tabs[2]:
    st.subheader("Test Connections")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Test UW"):
            ok, msg = uw.test_connection()
            st.success(msg) if ok else st.error(msg)

    with c2:
        if st.button("Test Polygon"):
            ok, msg = polygon.test_connection()
            st.success(msg) if ok else st.error(msg)

    with c3:
        if st.button("Test EODHD"):
            ok, msg = eodhd.test_connection()
            st.success(msg) if ok else st.error(msg)
