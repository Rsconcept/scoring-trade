#!/usr/bin/env python3
"""
Streamlit v3.1 Options Flow Scanner (Replay Mode + Local IV History)
====================================================================
What’s new (requested):
- Uses EODHD OPTIONS CHAIN snapshot IV (current IV per contract)
- Stores IV DAILY in a local JSON file (iv_history_store.json)
- Computes IV ramp from YOUR stored history (no more EODHD 422 dependency)
- When IV ramp is detected from local history, max score cap unlocks (up to 12)

Notes:
- Streamlit Community Cloud has an ephemeral filesystem (files can reset on redeploy).
  Replay snapshots + IV store are best-effort. If you want persistence, we can add
  Google Drive / S3 / Supabase later.

Deploy:
- app.py in repo root
- requirements.txt:
    streamlit
    requests

Secrets:
UW_TOKEN, POLYGON_API_KEY, EODHD_API_KEY
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

# -------------------- Constants --------------------

CT_OFFSET = -6  # Central Time offset
MAX_PENDING_TRADES = 50

PENDING_TRADES_FILE = "pending_trades.json"
INVERSE_SIGNALS_FILE = "inverse_signals.json"
VALIDATED_TRADES_FILE = "validated_trades.json"

SNAPSHOT_FILE = "last_uw_flows.json"

# NEW: Local IV store file
IV_STORE_FILE = "iv_history_store.json"

EXCLUDED_TICKERS_DEFAULT = {
    # Indexes
    "SPX", "SPXW", "NDX", "VIX", "RUT", "DJX", "XSP", "OEX",
    # Index ETFs
    "SPY", "QQQ", "IWM", "DIA",
    # Sector ETFs
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
    """Stable contract key for storing IV history."""
    return f"{ticker.upper()}|{expiry}|{option_type.lower()}|{float(strike):.2f}"


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
    """
    Stores IV history per contract in JSON:
    {
      "AAPL|2026-03-20|call|200.00": [
        {"date":"2026-02-10","iv":45.2},
        {"date":"2026-02-11","iv":48.1}
      ],
      ...
    }
    """
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

        # If today's entry exists, replace; else append
        replaced = False
        for r in rows:
            if isinstance(r, dict) and r.get("date") == t:
                r["iv"] = float(iv_value)
                replaced = True
                break
        if not replaced:
            rows.append({"date": t, "iv": float(iv_value)})

        # Keep last 90 entries
        rows = [r for r in rows if isinstance(r, dict) and r.get("date") and r.get("iv") is not None]
        rows.sort(key=lambda r: r["date"])
        rows = rows[-90:]

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

    def detect_ramp(self, key: str, lookback_days: int = 3, require_strict: bool = True) -> Tuple[bool, List[Tuple[str, float]]]:
        """
        Ramp = last N daily IV values are increasing (strict by default).
        Returns (is_ramping, last_points).
        """
        hist = self.get_history(key)
        if len(hist) < lookback_days:
            return False, []
        dates_sorted = sorted(hist.keys())
        last_dates = dates_sorted[-lookback_days:]
        pts = [(d, float(hist[d])) for d in last_dates]

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


# -------------------- API Clients --------------------

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
        status, data, err = http_get(
            f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/prev",
            params={"apiKey": self.api_key},
        )
        if status == 200 and data:
            results = data.get("results") or []
            if results:
                return safe_float(results[0].get("c", 0.0)), "Polygon prev close"
        return 0.0, f"Error: {err or 'no data'}"

    def get_spot_at_time(self, ticker: str, timestamp_ct: str) -> Tuple[float, str]:
        ticker = ticker.strip().upper()
        try:
            if not timestamp_ct or "CT" not in timestamp_ct:
                return self.get_previous_close(ticker)

            parts = timestamp_ct.replace(" CT", "").strip()
            dt_ct = datetime.strptime(parts, "%Y-%m-%d %I:%M:%S %p")
            trade_date = dt_ct.strftime("%Y-%m-%d")

            url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{trade_date}/{trade_date}"
            status, data, _ = http_get(url, params={"apiKey": self.api_key, "limit": 500})

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

            return self.get_previous_close(ticker)
        except Exception as e:
            return 0.0, f"Error: {e}"

    def get_price_history(self, ticker: str, days: int = 30) -> List[Dict[str, Any]]:
        ticker = ticker.strip().upper()
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
    """
    EODHD chain snapshot used for current IV.
    We keep marketplace attempt out of the critical path; the scanner mainly uses chain IV.
    """
    BASE_URL = "https://eodhd.com/api"

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    def test_connection(self) -> Tuple[bool, str]:
        status, data, err = http_get(
            f"{self.BASE_URL}/eod/AAPL.US",
            params={"api_token": self.api_key, "fmt": "json", "from": "2025-01-01", "limit": 1},
        )
        if status == 200 and data:
            return True, "OK (EODHD)"
        return False, err or "Failed"

    def get_iv_from_chain(self, ticker: str, strike: float, expiry: str, option_type: str) -> float:
        """Return current IV% for a contract if found; else 0."""
        ticker = ticker.strip().upper()
        option_type = option_type.lower().strip()
        if not self.api_key:
            return 0.0
        try:
            url = f"{self.BASE_URL}/options/{ticker}.US"
            status, data, _ = http_get(url, params={"api_token": self.api_key, "fmt": "json"})
            if status != 200 or not data or not isinstance(data, dict):
                return 0.0

            for exp_key, chain in data.items():
                if expiry not in str(exp_key):
                    continue
                if not isinstance(chain, dict):
                    continue

                options_list = chain.get(option_type + "s", [])
                if isinstance(options_list, dict):
                    options_list = list(options_list.values())
                if not isinstance(options_list, list):
                    continue

                for opt in options_list:
                    if not isinstance(opt, dict):
                        continue
                    opt_strike = safe_float(opt.get("strike", 0))
                    if abs(opt_strike - strike) < 0.5:
                        iv = safe_float(opt.get("impliedVolatility", 0))
                        if iv > 0:
                            # Normalize: if it's 0.xx, treat as fraction
                            if iv < 1:
                                iv *= 100
                            return float(iv)
            return 0.0
        except Exception:
            return 0.0


# -------------------- Enrichment + Ladder --------------------

class DataEnricher:
    def __init__(
        self,
        uw: UnusualWhalesAPI,
        polygon: PolygonAPI,
        eodhd: EODHDAPI,
        iv_store: LocalIVStore,
        iv_lookback_days: int = 3,
        iv_require_strict: bool = True,
    ):
        self.uw = uw
        self.polygon = polygon
        self.eodhd = eodhd
        self.iv_store = iv_store
        self.iv_lookback_days = iv_lookback_days
        self.iv_require_strict = iv_require_strict

    def enrich_trade(self, raw_flow: Dict[str, Any], use_iv: bool = True) -> Dict[str, Any]:
        ticker = str(raw_flow.get("ticker", "")).upper()
        strike = safe_float(raw_flow.get("strike", 0))
        option_type = str(raw_flow.get("option_type", "call")).lower()
        expiry = str(raw_flow.get("expiry", ""))
        timestamp = str(raw_flow.get("start_time", ""))

        timestamp_ct = to_central_time(timestamp)

        # Spot price
        spot, spot_source = self.polygon.get_spot_at_time(ticker, timestamp_ct)

        # Strike distance / OTM
        if spot > 0:
            strike_dist_pct = abs(strike - spot) / spot * 100.0
            is_otm = (strike > spot) if option_type == "call" else (strike < spot)
        else:
            strike_dist_pct = 0.0
            is_otm = True

        # Premium + Ask%
        total_prem = (
            safe_float(raw_flow.get("total_ask_side_prem", 0))
            + safe_float(raw_flow.get("total_bid_side_prem", 0))
            + safe_float(raw_flow.get("total_mid_side_prem", 0))
            + safe_float(raw_flow.get("total_no_side_prem", 0))
        )
        ask_prem = safe_float(raw_flow.get("total_ask_side_prem", 0))
        ask_pct = (ask_prem / total_prem * 100.0) if total_prem > 0 else 0.0

        # Volume/OI
        volume = safe_int(raw_flow.get("total_size", 0))
        oi = safe_int(raw_flow.get("open_interest", 0))
        vol_oi_ratio = (volume / oi) if oi > 0 else 999.0

        # Premium % heuristic
        denom = (spot * 100.0 * max(volume, 1)) if spot > 0 else 0.0
        premium_pct = (total_prem / denom * 100.0) if denom > 0 else 0.0

        dte = calculate_dte(expiry)

        # S/R + wick
        sr_data = self.polygon.calculate_support_resistance(ticker, strike)

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

        # NEW: Current IV from chain + local IV history/ramp
        ckey = contract_key(ticker, expiry, option_type, strike)
        current_iv = 0.0
        if use_iv and self.eodhd.api_key:
            current_iv = self.eodhd.get_iv_from_chain(ticker, strike, expiry, option_type)
            if current_iv > 0:
                self.iv_store.upsert_today(ckey, current_iv)

        local_iv_history = self.iv_store.get_history(ckey)  # date->iv
        iv_ramping, ramp_points = self.iv_store.detect_ramp(
            ckey, lookback_days=self.iv_lookback_days, require_strict=self.iv_require_strict
        )

        # Clean exception (as described)
        clean_exception = (
            ask_pct >= 70 and vol_oi_ratio > 1 and strike_dist_pct <= 7 and 2.5 <= premium_pct <= 5.0
        )

        return {
            "ticker": ticker,
            "strike": strike,
            "option_type": option_type,
            "expiry": expiry,
            "entry_timestamp": timestamp_ct,
            "spot": spot,
            "spot_source": spot_source,
            "strike_dist_pct": strike_dist_pct,
            "is_otm": is_otm,
            "total_premium": total_prem,
            "premium_pct": premium_pct,
            "volume": volume,
            "open_interest": oi,
            "vol_oi_ratio": vol_oi_ratio,
            "ask_pct": ask_pct,
            "dte": dte,
            "wick_triggered": bool(sr_data.get("wick_triggered", False)),
            "support": safe_float(sr_data.get("support", 0)),
            "resistance": safe_float(sr_data.get("resistance", 0)),
            # NEW fields
            "contract_key": ckey,
            "current_iv": float(current_iv),
            "iv_history_local": local_iv_history,
            "iv_ramping": bool(iv_ramping),
            "iv_ramp_points": ramp_points,  # list of (date, iv)
            "iv_ramp_lookback_days": int(self.iv_lookback_days),
            "days_to_earnings": days_to_er,
            "clean_exception": clean_exception,
            "has_sweep": bool(raw_flow.get("is_sweep", False)),
            "ladder_role": "isolated",
            "related_strikes": [],
            "category_tags": [],
            "_raw": raw_flow,
        }


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


# -------------------- Scoring --------------------

class V31ScoringEngine:
    def score(self, record: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        factors: List[str] = []
        penalties: List[str] = []
        record["category_tags"] = record.get("category_tags") or []

        # Premium % heuristic
        prem_pct = safe_float(record.get("premium_pct", 0))
        if 2.5 <= prem_pct <= 5.0:
            score += 2
            factors.append(f"Premium {prem_pct:.1f}% (+2)")
        elif 1.0 <= prem_pct < 2.5:
            score += 1
            factors.append(f"Premium {prem_pct:.1f}% (+1)")
        elif prem_pct < 1.0:
            score -= 2
            penalties.append(f"Ultra-low premium {prem_pct:.2f}% (-2)")
        else:
            factors.append(f"Excessive premium {prem_pct:.1f}% (0)")

        # Strike distance
        dist = safe_float(record.get("strike_dist_pct", 0))
        if dist <= 7:
            score += 2
            factors.append(f"Strike {dist:.1f}% OTM (+2)")
        elif dist <= 15:
            factors.append(f"Strike {dist:.1f}% OTM (0)")
        else:
            score -= 2
            penalties.append(f"Strike {dist:.1f}% deep OTM (-2)")

        # DTE
        dte = safe_int(record.get("dte", 0))
        if 7 <= dte <= 21:
            score += 1
            factors.append(f"DTE {dte}d (+1)")
        elif dte <= 1:
            score -= 1
            penalties.append("0-1 DTE (-1)")

        # Execution side
        ask_pct = safe_float(record.get("ask_pct", 0))
        if ask_pct >= 70:
            score += 1
            factors.append(f"Ask {ask_pct:.0f}% (+1)")
        elif 0 < ask_pct < 30:
            score -= 2
            penalties.append(f"Bid/mid heavy (Ask {ask_pct:.0f}%) (-2)")
        else:
            factors.append("Execution side unknown/neutral (0)")

        # Volume vs OI
        vol_oi = safe_float(record.get("vol_oi_ratio", 0))
        if vol_oi >= 2:
            score += 2
            factors.append(f"Vol/OI {vol_oi:.1f}x (+2)")
        elif vol_oi >= 1:
            score += 1
            factors.append(f"Vol/OI {vol_oi:.1f}x (+1)")

        # Wick rule
        if bool(record.get("wick_triggered", False)):
            score -= 2
            penalties.append("Wick reversal strike (-2)")

        # NEW: IV ramp (from local history)
        iv_ramping = bool(record.get("iv_ramping", False))
        if iv_ramping:
            score += 1
            pts = record.get("iv_ramp_points") or []
            if isinstance(pts, list) and pts:
                factors.append(f"IV ramp (local) (+1) {pts}")
            else:
                factors.append("IV ramp (local) (+1)")

        # Ladder/cluster
        ladder_role = str(record.get("ladder_role", "isolated")).lower()
        if ladder_role in ("anchor", "specleg", "ladder"):
            score += 1
            factors.append("Ladder/cluster (+1)")
        else:
            score -= 1
            penalties.append("Isolated (-1)")

        # Support penalty for puts
        if str(record.get("option_type", "")).lower() == "put":
            strike = safe_float(record.get("strike", 0))
            support = safe_float(record.get("support", 0))
            if strike > support and support > 0:
                score -= 1
                penalties.append("Put above support (-1)")

        # Catalyst bonus (2–10 days)
        days_to_er = record.get("days_to_earnings")
        if isinstance(days_to_er, int) and 2 <= days_to_er <= 10:
            score += 1
            factors.append(f"Catalyst {days_to_er}d (+1)")

        # Cap logic (unlocks if local IV ramp is true)
        if iv_ramping:
            max_score = 12
        elif bool(record.get("clean_exception", False)):
            max_score = 7
        else:
            max_score = 6

        final_score = min(score, max_score)

        # Verdict labels
        if final_score >= 8:
            verdict = "HIGH CONVICTION"
            record["category_tags"].append("HighConviction")
        elif final_score >= 7:
            verdict = "TRADEABLE"
            record["category_tags"].append("Tradeable")
        elif final_score >= 6:
            verdict = "MODERATE"
            record["category_tags"].append("Moderate")
        elif final_score >= 5:
            verdict = "WATCHLIST"
            record["category_tags"].append("Watchlist")
        else:
            verdict = "TRAP / SKIP"
            record["category_tags"].append("Trap")

        # Tags
        if bool(record.get("has_sweep", False)):
            record["category_tags"].append("Sweep")
        if vol_oi >= 10:
            record["category_tags"].append("LonelyWhale")
        if isinstance(days_to_er, int) and days_to_er <= 10:
            record["category_tags"].append("PreER")
        if safe_float(record.get("current_iv", 0)) > 0:
            record["category_tags"].append("HasIV")

        record["predictive_score"] = int(final_score)
        record["max_score"] = int(max_score)
        record["score_factors"] = factors
        record["score_penalties"] = penalties
        record["verdict"] = verdict
        return record


# -------------------- Queues --------------------

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

    def add_validated(self, trade: Dict[str, Any]) -> None:
        q = self.load_queue(self.validated_file)
        q.append(trade)
        self.save_queue(self.validated_file, q)


# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="v3.1 Options Flow Scanner", layout="wide")
st.title("v3.1 Options Flow Scanner (Replay + Local IV Ramp)")
st.caption("UW + Polygon + EODHD chain IV • Local IV history • Live/Replay toggle • JSON queues")

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
    st.header("API Keys (Secrets recommended)")
    uw_token = st.text_input("Unusual Whales Token", value=os.getenv("UW_TOKEN", ""), type="password")
    polygon_key = st.text_input("Polygon API Key", value=os.getenv("POLYGON_API_KEY", ""), type="password")
    eodhd_key = st.text_input("EODHD API Key", value=os.getenv("EODHD_API_KEY", ""), type="password")

    st.divider()
    st.subheader("Snapshot Tools")
    uploaded = st.file_uploader("Upload snapshot JSON (optional)", type=["json"])
    if uploaded is not None:
        try:
            up = json.loads(uploaded.read().decode("utf-8"))
            if isinstance(up, dict) and isinstance(up.get("flows"), list):
                snapshot.save(up["flows"])
                st.success("Uploaded snapshot saved.")
            elif isinstance(up, list):
                snapshot.save(up)
                st.success("Uploaded snapshot saved.")
            else:
                st.error("Snapshot format not recognized.")
        except Exception as e:
            st.error(f"Upload error: {e}")

    if snapshot.exists():
        st.download_button(
            "Download current snapshot",
            data=snapshot.raw_text(),
            file_name="last_uw_flows.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Local IV Store")
    st.caption("This is how we build IV ramp for FREE (store IV daily).")
    if st.button("Download IV store", use_container_width=True):
        st.download_button(
            "Click to download iv_history_store.json",
            data=iv_store.raw_text(),
            file_name="iv_history_store.json",
            mime="application/json",
            use_container_width=True,
        )
    if st.button("Reset IV store (danger)", type="secondary", use_container_width=True):
        iv_store.reset()
        st.warning("IV store reset.")

    st.divider()
    st.subheader("IV Ramp Settings")
    iv_lookback_days = st.slider("Ramp lookback days", 3, 10, 3, 1)
    iv_strict = st.checkbox("Require strictly increasing IV", value=True)

    st.divider()
    st.subheader("Scan Controls")
    use_iv = st.checkbox("Use IV (EODHD chain) + store locally", value=True)
    limit = st.slider("UW flow alerts limit (Live only)", min_value=10, max_value=250, value=200, step=10)

    min_premium = st.number_input("Min premium ($)", min_value=0, value=25_000, step=5_000)
    min_size = st.number_input("Min size (contracts)", min_value=0, value=0, step=100)
    min_vol_oi = st.number_input("Min Vol/OI", min_value=0.0, value=1.0, step=0.1)

    require_vol_gt_oi = st.checkbox("Require Vol > OI", value=False)
    exclude_indices = st.checkbox("Exclude indices + major ETFs (SPX/SPY/QQQ/etc.)", value=True)

    st.divider()
    st.subheader("Ladder Settings")
    ladder_minutes = st.slider("Ladder time window (minutes)", 15, 240, 90, 15)
    ladder_min_strikes = st.slider("Min unique strikes to call a ladder", 2, 6, 3, 1)

    st.divider()
    st.subheader("Queues")
    pending_path = st.text_input("Pending file", value=PENDING_TRADES_FILE)
    inverse_path = st.text_input("Inverse file", value=INVERSE_SIGNALS_FILE)
    validated_path = st.text_input("Validated file", value=VALIDATED_TRADES_FILE)

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
)
scorer = V31ScoringEngine()
ladder = LadderDetector(uw)
queue = QueueManager(pending_path, inverse_path, validated_path)

tabs = st.tabs(["Scan", "Manual Score", "Validate T+1", "Queues", "Connections"])


def get_source_flows() -> Tuple[List[Dict[str, Any]], str]:
    if data_mode:
        return snapshot.load(), "Replay (snapshot)"
    return uw.get_flows(limit=limit), "Live (UW API)"


# -------------------- Scan --------------------
with tabs[0]:
    st.subheader("Run Scanner")

    colA, colB = st.columns([1, 2], gap="large")
    with colA:
        run = st.button("Run scan", type="primary", use_container_width=True)
        st.caption("Market closed? Turn ON Replay Mode.")
    with colB:
        st.markdown(
            """
- Pulls UW flows (Live or Snapshot)
- Filters: min premium + exclusions + min size + min vol/oi (+ optional vol>OI)
- Enriches: Polygon spot + S/R + wick + (optional) EODHD chain IV
- Stores IV daily locally and detects IV ramp from YOUR history
- Scores v3.1 and writes queues:
  - pending (score >= 5), inverse (score <= -3)
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
                st.warning(f"No flows from {src}. If Replay Mode is ON, run Live once to create a snapshot.")
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

                        # Enrich (IV optional)
                        enriched = enricher.enrich_trade(f, use_iv=bool(use_iv and eodhd_key))

                        # Ladder optional (needs UW token)
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
                                "Dist%": round(safe_float(r.get("strike_dist_pct", 0.0)), 2),
                                "Premium$": round(safe_float(r.get("total_premium", 0.0))),
                                "Prem%": round(safe_float(r.get("premium_pct", 0.0)), 2),
                                "Ask%": round(safe_float(r.get("ask_pct", 0.0)), 1),
                                "Vol/OI": round(safe_float(r.get("vol_oi_ratio", 0.0)), 2),
                                "IV%": round(safe_float(r.get("current_iv", 0.0)), 2),
                                "IVRamp": bool(r.get("iv_ramping", False)),
                                "Wick": bool(r.get("wick_triggered", False)),
                                "Ladder": (r.get("ladder_role") != "isolated"),
                                "Score": r.get("predictive_score"),
                                "Max": r.get("max_score"),
                                "Verdict": r.get("verdict"),
                                "Tags": ", ".join(r.get("category_tags", [])),
                            }
                        )
                    st.dataframe(table, use_container_width=True, hide_index=True)

                    st.divider()
                    st.subheader("Details")
                    for r in results[:60]:
                        header = (
                            f"{r['ticker']} {str(r['option_type']).upper()} ${r['strike']} {r['expiry']} • "
                            f"Score {r['predictive_score']}/{r['max_score']} • {r['verdict']}"
                        )
                        with st.expander(header, expanded=False):
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.write("**Entry (CT)**", r.get("entry_timestamp"))
                                st.write("**Spot**", round(safe_float(r.get("spot", 0)), 2))
                                st.write("**Spot Source**", r.get("spot_source"))
                                st.write("**DTE**", r.get("dte"))
                                st.write("**Contract Key**", r.get("contract_key"))
                            with c2:
                                st.write("**Premium**", pretty_money(safe_float(r.get("total_premium", 0))))
                                st.write("**Prem %**", f"{safe_float(r.get('premium_pct', 0)):.2f}%")
                                st.write("**Ask %**", f"{safe_float(r.get('ask_pct', 0)):.1f}%")
                                st.write("**Vol/OI**", f"{safe_float(r.get('vol_oi_ratio', 0)):.2f}x")
                                st.write("**Sweep**", bool(r.get("has_sweep", False)))
                            with c3:
                                st.write("**IV (current)**", f"{safe_float(r.get('current_iv', 0)):.2f}%")
                                st.write("**IV Ramp**", bool(r.get("iv_ramping", False)))
                                st.write("**IV Ramp Points**", r.get("iv_ramp_points"))
                                st.write("**Earnings (days)**", r.get("days_to_earnings"))
                                st.write("**Ladder**", r.get("ladder_role", "isolated"))
                                if r.get("related_strikes"):
                                    st.write("**Related strikes**", r.get("related_strikes"))

                            st.write("**Factors**")
                            st.write("\n".join([f"• {x}" for x in r.get("score_factors", [])]) or "—")
                            st.write("**Penalties**")
                            st.write("\n".join([f"• {x}" for x in r.get("score_penalties", [])]) or "—")

                            st.write("**Local IV history (date -> IV%)**")
                            st.json(r.get("iv_history_local", {}))

                            st.write("**Raw UW (debug)**")
                            st.json(r.get("_raw", {}))


# -------------------- Manual Score --------------------
with tabs[1]:
    st.subheader("Manual Score (paste one trade JSON)")

    sample = {
        "ticker": "AAPL",
        "strike": 200,
        "option_type": "call",
        "expiry": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
        "start_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_size": 5000,
        "open_interest": 1000,
        "total_ask_side_prem": 1_200_000,
        "total_bid_side_prem": 0,
        "total_mid_side_prem": 0,
        "total_no_side_prem": 0,
        "is_sweep": True,
    }

    raw_text = st.text_area("Trade JSON", value=json.dumps(sample, indent=2), height=260)

    do_enrich = st.checkbox("Enrich using APIs", value=True)
    do_ladder = st.checkbox("Run ladder detection", value=False)
    do_iv = st.checkbox("Use IV + store locally", value=True)

    if st.button("Score this trade", type="primary"):
        try:
            raw_flow = json.loads(raw_text)
            if not isinstance(raw_flow, dict):
                raise ValueError("JSON must be an object")

            if do_enrich:
                enriched = enricher.enrich_trade(raw_flow, use_iv=bool(do_iv and eodhd_key))
                if do_ladder:
                    is_l, rel = ladder.detect(
                        ticker=enriched["ticker"],
                        target_strike=enriched["strike"],
                        option_type=enriched["option_type"],
                        expiry=enriched["expiry"],
                        recent_minutes=int(ladder_minutes),
                        min_unique_strikes=int(ladder_min_strikes),
                    )
                    if is_l:
                        enriched["ladder_role"] = "ladder"
                        enriched["related_strikes"] = rel

                scored = scorer.score(enriched)
                st.success(f"Score: {scored['predictive_score']} / max {scored['max_score']} • {scored['verdict']}")
                st.json(scored)
            else:
                scored = scorer.score(raw_flow)
                st.json(scored)

        except Exception as e:
            st.error(f"Error: {e}")


# -------------------- Validate T+1 --------------------
with tabs[2]:
    st.subheader("Validate T+1 (manual inputs)")

    pending = queue.load_queue(pending_path)
    if not pending:
        st.info("No pending trades. Run a scan first.")
    else:
        choices = [
            f"{i}: {t.get('ticker')} {str(t.get('option_type','')).upper()} {t.get('strike')} {t.get('expiry')} • score {t.get('predictive_score')}"
            for i, t in enumerate(pending)
        ]
        sel = st.selectbox("Select pending trade", choices, index=0)
        idx = int(sel.split(":")[0])
        trade = pending[idx]

        st.json(
            {k: trade.get(k) for k in [
                "ticker", "option_type", "strike", "expiry", "entry_timestamp",
                "predictive_score", "max_score", "verdict", "premium_pct", "ask_pct",
                "volume", "open_interest", "current_iv", "iv_ramping", "contract_key"
            ]}
        )

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            prior_oi = st.number_input("Prior-day OI", min_value=0, value=int(trade.get("open_interest", 0)), step=1)
            t1_oi = st.number_input("T+1 OI", min_value=0, value=int(trade.get("open_interest", 0)), step=1)
        with c2:
            iv_entry = st.number_input("Entry-day IV (%)", min_value=0.0, value=float(trade.get("current_iv", 0.0)), step=0.5)
            iv_t1 = st.number_input("T+1 IV (%)", min_value=0.0, value=0.0, step=0.5)
        with c3:
            roll_override = st.checkbox("Roll override context (manual)", value=False)

        if st.button("Apply T+1 validation + archive", type="primary"):
            t = dict(trade)
            vol = safe_int(t.get("volume", 0))
            oi_change = max(int(t1_oi - prior_oi), 0)
            oi_conv_pct = (oi_change / max(vol, 1)) * 100.0

            t["prior_oi_tminus1"] = int(prior_oi)
            t["tplus1_oi"] = int(t1_oi)
            t["oi_change_tplus1"] = int(oi_change)
            t["oi_change_pct_of_volume"] = round(oi_conv_pct, 2)
            t["iv_entry_manual"] = float(iv_entry)
            t["iv_tplus1_manual"] = float(iv_t1)

            t.setdefault("validation_notes", [])
            delta = 0

            if iv_entry > 0 and iv_t1 > 0:
                if iv_t1 > iv_entry:
                    delta += 1
                    t["validation_notes"].append("T+1 IV up (+1)")
                elif iv_t1 < iv_entry:
                    delta -= 1
                    t["validation_notes"].append("T+1 IV down (-1)")

            if oi_conv_pct < 10.0:
                t["validation_notes"].append("OI conversion <10% of volume (trap flag)")
            elif oi_conv_pct >= 50.0:
                t["validation_notes"].append("Strong OI conversion (>=50% of volume)")
            else:
                t["validation_notes"].append("Moderate OI conversion (10–50% of volume)")

            if roll_override:
                delta += 1
                t["validation_notes"].append("Roll override continuation (+1)")

            pred = safe_int(t.get("predictive_score", 0))
            t["validated_score"] = int(pred + delta)

            vs = t["validated_score"]
            if vs >= 8:
                t["validated_verdict"] = "HIGH CONVICTION"
            elif vs >= 7:
                t["validated_verdict"] = "TRADEABLE"
            elif vs >= 6:
                t["validated_verdict"] = "MODERATE"
            elif vs >= 5:
                t["validated_verdict"] = "WATCHLIST"
            else:
                t["validated_verdict"] = "TRAP / SKIP"

            queue.add_validated(t)
            pending.pop(idx)
            queue.save_queue(pending_path, pending)

            st.success(f"Archived. Validated score: {t['validated_score']} • {t['validated_verdict']}")
            st.json(t)


# -------------------- Queues --------------------
with tabs[3]:
    st.subheader("Queues")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("### Pending")
        pend = queue.load_queue(pending_path)
        st.write(f"{len(pend)} items")
        st.dataframe(
            [{"Ticker": t.get("ticker"), "Type": t.get("option_type"), "Strike": t.get("strike"),
              "Expiry": t.get("expiry"), "Score": t.get("predictive_score"), "Verdict": t.get("verdict")}
             for t in pend[-200:]],
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.write("### Inverse")
        inv = queue.load_queue(inverse_path)
        st.write(f"{len(inv)} items")
        st.dataframe(
            [{"Ticker": t.get("ticker"), "Type": t.get("option_type"), "Strike": t.get("strike"),
              "Expiry": t.get("expiry"), "Score": t.get("predictive_score"), "Verdict": t.get("verdict")}
             for t in inv[-200:]],
            use_container_width=True,
            hide_index=True,
        )

    with c3:
        st.write("### Validated")
        val = queue.load_queue(validated_path)
        st.write(f"{len(val)} items")
        st.dataframe(
            [{"Ticker": t.get("ticker"), "Type": t.get("option_type"), "Strike": t.get("strike"),
              "Expiry": t.get("expiry"), "Pred": t.get("predictive_score"),
              "Val": t.get("validated_score"), "Val Verdict": t.get("validated_verdict")}
             for t in val[-200:]],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    if st.button("Reset all queues (danger)", type="secondary"):
        queue.save_queue(pending_path, [])
        queue.save_queue(inverse_path, [])
        queue.save_queue(validated_path, [])
        st.warning("Queues cleared.")


# -------------------- Connections --------------------
with tabs[4]:
    st.subheader("Test Connections")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Test UW"):
            ok, msg = uw.test_connection()
            st.success(msg) if ok else st.error(msg)

    with col2:
        if st.button("Test Polygon"):
            ok, msg = polygon.test_connection()
            st.success(msg) if ok else st.error(msg)

    with col3:
        if st.button("Test EODHD"):
            ok, msg = eodhd.test_connection()
            st.success(msg) if ok else st.error(msg)

    st.divider()
    st.markdown(
        """
**How IV ramp works now (FREE)**
- Each time you scan, the app pulls **current IV** from EODHD chain for each contract found
- It stores one IV value per day in `iv_history_store.json`
- After you have at least **3 days** saved for a contract, the app can detect **IV ramp**
- When ramp is detected, max score cap unlocks (up to 12)

Tip: run at least 1 scan per day during market hours to build the IV history faster.
"""
    )
