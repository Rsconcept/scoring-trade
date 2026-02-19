#!/usr/bin/env python3
"""
Streamlit v3.1 Options Flow Scanner (UW + Polygon + EODHD)
=========================================================
Implements the v3.1 flow scanner described in v31_Scanner_Complete_Package.pdf
- Fetch: Unusual Whales flow alerts
- Filter: exclude indices, require Vol > OI
- Enrich: Polygon spot-at-time + S/R + wick detection
- Enrich: EODHD IV history (falls back to chain snapshot due to known 422 blocker)
- Score: v3.1 scoring engine (11 rules) with IV ramp gate + clean exception
- Queue: pending trades / inverse signals / validated trades in JSON

Notes:
- EODHD marketplace historical IV endpoint is documented as returning HTTP 422 for all filters;
  therefore IV ramp detection usually cannot be computed from EODHD and scores will be capped
  at 6 unless Clean Exception applies (cap 7). This matches the package notes.
"""

from __future__ import annotations

import os
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

# ==================== DEFAULT CONFIG (override via sidebar/env) ====================

CT_OFFSET = -6  # Central Time offset from UTC (package default)
EXCLUDED_TICKERS_DEFAULT = {'SPX', 'SPXW', 'NDX', 'VIX', 'RUT', 'DJX', 'XSP', 'OEX'}

MAX_PENDING_TRADES = 50

PENDING_TRADES_FILE = "pending_trades.json"
INVERSE_SIGNALS_FILE = "inverse_signals.json"
VALIDATED_TRADES_FILE = "validated_trades.json"


# ==================== HELPERS ====================

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


def http_get(url: str, headers: Optional[Dict[str, str]] = None,
             params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Tuple[int, Optional[Any], str]:
    """HTTP GET returning (status_code, json_or_none, error_string)."""
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
    """Convert ISO timestamp to Central Time string like 'YYYY-MM-DD HH:MM:SS AM CT'."""
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


def ensure_json_file(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


# ==================== API CLIENTS ====================

class UnusualWhalesAPI:
    """Unusual Whales API client for flow data."""
    BASE_URL = "https://api.unusualwhales.com/api"

    def __init__(self, token: str):
        self.token = token.strip()
        self.headers = {
            "Accept": "application/json, text/plain",
            "Authorization": f"Bearer {self.token}" if self.token else ""
        }

    def test_connection(self) -> Tuple[bool, str]:
        status, data, err = http_get(
            f"{self.BASE_URL}/option-trades/flow-alerts",
            headers=self.headers,
            params={"limit": 3}
        )
        if status == 200 and data:
            return True, "OK (/option-trades/flow-alerts)"
        return False, err or "Failed"

    def get_flows(self, limit: int = 100) -> List[Dict[str, Any]]:
        status, data, err = http_get(
            f"{self.BASE_URL}/option-trades/flow-alerts",
            headers=self.headers,
            params={"limit": limit}
        )
        if status != 200 or not data:
            return []
        return data.get("data", []) or []

    def get_ticker_flow(self, ticker: str, limit: int = 50) -> List[Dict[str, Any]]:
        ticker = ticker.strip().upper()
        status, data, err = http_get(
            f"{self.BASE_URL}/stock/{ticker}/options-flow",
            headers=self.headers,
            params={"limit": limit}
        )
        if status == 200 and data:
            return data.get("data", []) or []
        return []

    def get_earnings(self, ticker: str) -> List[Dict[str, Any]]:
        ticker = ticker.strip().upper()
        status, data, err = http_get(
            f"{self.BASE_URL}/stock/{ticker}/earnings-history",
            headers=self.headers
        )
        if status == 200 and data:
            return data.get("data", []) or []
        return []


class PolygonAPI:
    """Polygon API client for spot and daily history."""
    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    def test_connection(self) -> Tuple[bool, str]:
        status, data, err = http_get(
            f"{self.BASE_URL}/v2/aggs/ticker/AAPL/prev",
            params={"apiKey": self.api_key}
        )
        if status == 200 and data and data.get("status") == "OK":
            return True, "OK (/v2/aggs/ticker/AAPL/prev)"
        return False, err or "Failed"

    def get_previous_close(self, ticker: str) -> Tuple[float, str]:
        ticker = ticker.strip().upper()
        status, data, err = http_get(
            f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/prev",
            params={"apiKey": self.api_key}
        )
        if status == 200 and data:
            results = data.get("results") or []
            if results:
                return safe_float(results[0].get("c", 0.0)), "Polygon prev close"
        return 0.0, f"Error: {err or 'no data'}"

    def get_spot_at_time(self, ticker: str, timestamp_ct: str) -> Tuple[float, str]:
        """
        Get spot price at/near trade time using minute bars; fallback to previous close.
        timestamp_ct like 'YYYY-MM-DD HH:MM:SS AM CT'
        """
        ticker = ticker.strip().upper()
        try:
            if not timestamp_ct or "CT" not in timestamp_ct:
                return self.get_previous_close(ticker)

            parts = timestamp_ct.replace(" CT", "").strip()
            dt = datetime.strptime(parts, "%Y-%m-%d %I:%M:%S %p")
            trade_date = dt.strftime("%Y-%m-%d")
            url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{trade_date}/{trade_date}"

            status, data, err = http_get(url, params={"apiKey": self.api_key, "limit": 500})
            if status == 200 and data and data.get("results"):
                bars = data["results"]
                target_min = dt.hour * 60 + dt.minute

                closest = None
                min_diff = 10_000
                for bar in bars:
                    bar_ts = safe_float(bar.get("t", 0)) / 1000.0
                    bar_dt_utc = datetime.utcfromtimestamp(bar_ts)
                    bar_ct = bar_dt_utc + timedelta(hours=CT_OFFSET)
                    bar_min = bar_ct.hour * 60 + bar_ct.minute
                    diff = abs(bar_min - target_min)
                    if diff < min_diff:
                        min_diff = diff
                        closest = bar
                if closest and min_diff <= 5:
                    return safe_float(closest.get("c", 0.0)), "Polygon intraday"
            return self.get_previous_close(ticker)
        except Exception as e:
            return 0.0, f"Error: {e}"

    def get_price_history(self, ticker: str, days: int = 30) -> List[Dict[str, Any]]:
        ticker = ticker.strip().upper()
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            url = (f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/"
                   f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}")
            status, data, err = http_get(url, params={"apiKey": self.api_key, "limit": days})
            if status == 200 and data:
                out = []
                for r in (data.get("results") or []):
                    out.append({
                        "date": datetime.fromtimestamp(safe_float(r.get("t", 0))/1000.0).strftime("%Y-%m-%d"),
                        "open": safe_float(r.get("o", 0)),
                        "high": safe_float(r.get("h", 0)),
                        "low": safe_float(r.get("l", 0)),
                        "close": safe_float(r.get("c", 0)),
                        "volume": safe_float(r.get("v", 0)),
                    })
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
            "wick_triggered": wick_triggered
        }


class EODHDAPI:
    """
    EODHD API for IV and options chain snapshot.

    Package notes:
    - Marketplace historical IV endpoint returns 422, so we fall back to options chain (current IV only).
    """
    BASE_URL = "https://eodhd.com/api"

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    def test_connection(self) -> Tuple[bool, str]:
        status, data, err = http_get(
            f"{self.BASE_URL}/eod/AAPL.US",
            params={"api_token": self.api_key, "fmt": "json", "from": "2025-01-01", "limit": 1}
        )
        if status == 200 and data:
            return True, "OK (/eod/AAPL.US)"
        return False, err or "Failed"

    def get_iv_history(self, ticker: str, strike: float, expiry: str, option_type: str = "call") -> Dict[str, float]:
        """
        Attempt marketplace endpoint, else fallback to chain snapshot.

        Returns dict date->IV(%).
        """
        ticker = ticker.strip().upper()
        option_type = option_type.lower().strip()
        try:
            url = f"{self.BASE_URL}/mp/unicornbay/options/eod"
            params = {
                "api_token": self.api_key,
                "fmt": "json",
                "filter[underlying_symbol]": ticker,
                "filter[type]": option_type,
            }
            status, data, err = http_get(url, params=params)
            if status == 200 and data:
                records = data.get("data", data) if isinstance(data, dict) else data
                iv_history: Dict[str, float] = {}
                if isinstance(records, list):
                    for record in records:
                        attrs = record.get("attributes", record) if isinstance(record, dict) else {}
                        rec_strike = safe_float(attrs.get("strike", 0))
                        if abs(rec_strike - strike) > 1.0:
                            continue
                        rec_expiry = str(attrs.get("exp_date", "")) or str(attrs.get("expiration", ""))
                        if rec_expiry != expiry:
                            continue
                        iv = attrs.get("volatility") or attrs.get("impliedVolatility") or 0
                        trade_date = str(attrs.get("tradetime", "")).split("T")[0]
                        if iv and trade_date:
                            iv_val = safe_float(iv)
                            if iv_val < 10:
                                iv_val *= 100
                            iv_history[trade_date] = iv_val
                if iv_history:
                    return iv_history
            # fallback
            return self._get_iv_from_chain(ticker, strike, expiry, option_type)
        except Exception:
            return self._get_iv_from_chain(ticker, strike, expiry, option_type)

    def _get_iv_from_chain(self, ticker: str, strike: float, expiry: str, option_type: str) -> Dict[str, float]:
        """Fallback: current IV from /options endpoint."""
        ticker = ticker.strip().upper()
        option_type = option_type.lower().strip()
        try:
            url = f"{self.BASE_URL}/options/{ticker}.US"
            status, data, err = http_get(url, params={"api_token": self.api_key, "fmt": "json"})
            if status != 200 or not data or not isinstance(data, dict):
                return {}

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
                            if iv < 1:
                                iv *= 100
                            today = datetime.now().strftime("%Y-%m-%d")
                            return {today: iv}
            return {}
        except Exception:
            return {}


# ==================== ENRICHMENT + LADDER ====================

class DataEnricher:
    """Combines UW + Polygon + EODHD into a trade record."""
    def __init__(self, uw: UnusualWhalesAPI, polygon: PolygonAPI, eodhd: EODHDAPI):
        self.uw = uw
        self.polygon = polygon
        self.eodhd = eodhd

    def enrich_trade(self, raw_flow: Dict[str, Any], fetch_iv: bool = True) -> Dict[str, Any]:
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

        # Premium calc (package uses total_ask/bid/mid/no prem sum)
        total_prem = (
            safe_float(raw_flow.get("total_ask_side_prem", 0)) +
            safe_float(raw_flow.get("total_bid_side_prem", 0)) +
            safe_float(raw_flow.get("total_mid_side_prem", 0)) +
            safe_float(raw_flow.get("total_no_side_prem", 0))
        )

        # Ask %
        ask_prem = safe_float(raw_flow.get("total_ask_side_prem", 0))
        ask_pct = (ask_prem / total_prem * 100.0) if total_prem > 0 else 0.0

        # Volume/OI
        volume = safe_int(raw_flow.get("total_size", 0))
        oi = safe_int(raw_flow.get("open_interest", 0))
        vol_oi_ratio = (volume / oi) if oi > 0 else 999.0

        # Premium % (package logic uses spot * 100 * total_size; keep as-is for consistency)
        denom = (spot * 100.0 * max(volume, 1)) if spot > 0 else 0.0
        premium_pct = (total_prem / denom * 100.0) if denom > 0 else 0.0

        dte = calculate_dte(expiry)

        # S/R + wick
        sr_data = self.polygon.calculate_support_resistance(ticker, strike)

        # IV history
        iv_history: Dict[str, float] = {}
        iv_ramping = False
        if fetch_iv:
            iv_history = self.eodhd.get_iv_history(ticker, strike, expiry, option_type)
            if len(iv_history) >= 3:
                dates = sorted(iv_history.keys())
                vals = [iv_history[d] for d in dates[-3:]]
                iv_ramping = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

        # Earnings catalyst window (0-30 days; score bonus applied later for 2-10 days)
        earnings = self.uw.get_earnings(ticker)
        days_to_er: Optional[int] = None
        if earnings:
            today = datetime.now()
            for er in earnings:
                er_date_str = str(er.get("date", "")).strip()
                try:
                    er_date = datetime.strptime(er_date_str, "%Y-%m-%d")
                    delta = (er_date - today).days
                    if 0 <= delta <= 30:
                        days_to_er = delta
                        break
                except Exception:
                    continue

        # Clean exception (package uses these four)
        clean_exception = (
            ask_pct >= 70 and
            vol_oi_ratio > 1 and
            strike_dist_pct <= 7 and
            2.5 <= premium_pct <= 5.0
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
            "iv_history": iv_history,
            "iv_ramping": iv_ramping,
            "days_to_earnings": days_to_er,
            "clean_exception": clean_exception,
            "has_sweep": bool(raw_flow.get("is_sweep", False)),
            "ladder_role": "isolated",
            "category_tags": [],
            "_raw": raw_flow,  # keep for debugging in UI
        }


class LadderDetector:
    """Detects multi-strike ladder patterns (simple package implementation)."""
    def __init__(self, uw: UnusualWhalesAPI):
        self.uw = uw

    def detect(self, ticker: str, strike: float, option_type: str, expiry: str) -> Tuple[bool, List[float]]:
        flows = self.uw.get_ticker_flow(ticker, limit=50)
        if not flows:
            return False, []
        related: List[float] = []
        for f in flows:
            f_type = str(f.get("option_type", "")).lower()
            f_expiry = str(f.get("expiry", ""))
            f_strike = safe_float(f.get("strike", 0))
            if f_type == option_type and f_expiry == expiry and abs(f_strike - strike) > 1e-9:
                related.append(f_strike)
        if len(set(related)) >= 2:
            return True, sorted(set(related))
        return False, []


# ==================== SCORING ====================

class V31ScoringEngine:
    """v3.1 scoring engine (package implementation: 11 rules + cap + tags)."""

    def score(self, record: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        factors: List[str] = []
        penalties: List[str] = []
        record["category_tags"] = record.get("category_tags") or []

        # RULE 4: Premium % of Spot
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

        # RULE 5: Strike Distance
        dist = safe_float(record.get("strike_dist_pct", 0))
        if dist <= 7:
            score += 2
            factors.append(f"Strike {dist:.1f}% OTM (+2)")
        elif dist <= 15:
            factors.append(f"Strike {dist:.1f}% OTM (0)")
        else:
            score -= 2
            penalties.append(f"Strike {dist:.1f}% deep OTM (-2)")

        # RULE 6: DTE
        dte = safe_int(record.get("dte", 0))
        if 7 <= dte <= 21:
            score += 1
            factors.append(f"DTE {dte}d (+1)")
        elif dte <= 1:
            score -= 1
            penalties.append("0-1 DTE (-1)")

        # RULE 7: Execution Side (using ask % heuristic from UW)
        ask_pct = safe_float(record.get("ask_pct", 0))
        if ask_pct >= 70:
            score += 1
            factors.append(f"Ask {ask_pct:.0f}% (+1)")
        elif ask_pct < 30:
            score -= 2
            penalties.append("Bid/mid heavy (-2)")

        # RULE 8: Volume vs OI (package uses vol_oi_ratio)
        vol_oi = safe_float(record.get("vol_oi_ratio", 0))
        if vol_oi >= 2:
            score += 2
            factors.append(f"Vol/OI {vol_oi:.1f}x (+2)")
        elif vol_oi >= 1:
            score += 1
            factors.append(f"Vol/OI {vol_oi:.1f}x (+1)")

        # RULE 3: Wick Rule
        if bool(record.get("wick_triggered", False)):
            score -= 2
            penalties.append("Wick reversal strike (-2)")

        # RULE 9: IV ramp (gate input)
        iv_ramping = bool(record.get("iv_ramping", False))
        if iv_ramping:
            score += 1
            factors.append("IV ramp detected (+1)")

        # RULE 10: Ladder/cluster
        ladder_role = str(record.get("ladder_role", "isolated"))
        if ladder_role in ("anchor", "specleg", "ladder"):
            score += 1
            factors.append("Ladder/cluster (+1)")
        else:
            score -= 1
            penalties.append("Isolated (-1)")

        # RULE 11: Support penalty (puts)
        if str(record.get("option_type", "")).lower() == "put":
            strike = safe_float(record.get("strike", 0))
            support = safe_float(record.get("support", 0))
            if strike > support and support > 0:
                score -= 1
                penalties.append("Put above support (-1)")

        # Catalyst proximity bonus (package uses 2-10 days)
        days_to_er = record.get("days_to_earnings")
        if isinstance(days_to_er, int) and 2 <= days_to_er <= 10:
            score += 1
            factors.append(f"Catalyst {days_to_er}d (+1)")

        # IV ramp gate cap
        if iv_ramping:
            max_score = 12
        elif bool(record.get("clean_exception", False)):
            max_score = 7
        else:
            max_score = 6

        final_score = min(score, max_score)

        # Verdict + tags
        if final_score >= 8:
            verdict = "HIGH CONVICTION"
            record["category_tags"].append("HighConviction")
        elif final_score >= 7:
            verdict = "TRADEABLE"
            record["category_tags"].append("Tradeable")
        elif final_score >= 6:
            verdict = "MODERATE"
            record["category_tags"].append("Moderate")
        else:
            verdict = "TRAP / SKIP"
            record["category_tags"].append("Trap")

        if bool(record.get("has_sweep", False)):
            record["category_tags"].append("Sweep")
        if vol_oi >= 10:
            record["category_tags"].append("LonelyWhale")
        if isinstance(days_to_er, int) and days_to_er <= 10:
            record["category_tags"].append("PreER")

        record["predictive_score"] = final_score
        record["max_score"] = max_score
        record["score_factors"] = factors
        record["score_penalties"] = penalties
        record["verdict"] = verdict
        return record


# ==================== QUEUES ====================

class QueueManager:
    """JSON-based queue for pending / inverse / validated."""
    def __init__(self, pending_file: str, inverse_file: str, validated_file: str):
        self.pending_file = pending_file
        self.inverse_file = inverse_file
        self.validated_file = validated_file
        ensure_json_file(self.pending_file)
        ensure_json_file(self.inverse_file)
        ensure_json_file(self.validated_file)

    def save_queue(self, filepath: str, data: List[Dict[str, Any]]) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load_queue(self, filepath: str) -> List[Dict[str, Any]]:
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                x = json.load(f)
                return x if isinstance(x, list) else []
            except Exception:
                return []

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


# ==================== STREAMLIT APP ====================

st.set_page_config(page_title="v3.1 Options Flow Scanner", layout="wide")

st.title("v3.1 Options Flow Scanner (Streamlit)")
st.caption("UW + Polygon + EODHD • v3.1 scoring • JSON queues (pending/inverse/validated)")

with st.sidebar:
    st.header("Settings")

    uw_token = st.text_input(
        "Unusual Whales Token",
        value=os.getenv("UW_TOKEN", ""),
        type="password",
        help="Set env UW_TOKEN for GitHub/Streamlit deployments."
    )
    polygon_key = st.text_input(
        "Polygon API Key",
        value=os.getenv("POLYGON_API_KEY", ""),
        type="password",
        help="Set env POLYGON_API_KEY."
    )
    eodhd_key = st.text_input(
        "EODHD API Key",
        value=os.getenv("EODHD_API_KEY", ""),
        type="password",
        help="Set env EODHD_API_KEY."
    )

    st.divider()
    st.subheader("Scan Controls")
    mode = st.radio("Mode", ["Full Scan (with IV fetch)", "Quick Scan (skip IV)"], index=0)
    fetch_iv = (mode.startswith("Full"))

    limit = st.slider("UW flow alerts limit", min_value=10, max_value=200, value=100, step=10)

    min_premium = st.number_input("Min premium ($)", min_value=0, value=1_000_000, step=100_000)
    require_vol_gt_oi = st.checkbox("Require Vol > OI", value=True)
    exclude_indices = st.checkbox("Exclude index tickers (SPX/NDX/etc.)", value=True)

    st.divider()
    st.subheader("Queues")
    pending_path = st.text_input("Pending file", value=PENDING_TRADES_FILE)
    inverse_path = st.text_input("Inverse file", value=INVERSE_SIGNALS_FILE)
    validated_path = st.text_input("Validated file", value=VALIDATED_TRADES_FILE)

    st.divider()
    st.subheader("About IV ramp")
    st.info(
        "EODHD historical IV marketplace endpoint is documented as returning HTTP 422, "
        "so IV ramp detection usually falls back to chain snapshot (1 point) and scores "
        "are capped at 6 unless Clean Exception applies (cap 7)."
    )

# Instantiate components
uw = UnusualWhalesAPI(uw_token)
polygon = PolygonAPI(polygon_key)
eodhd = EODHDAPI(eodhd_key)
enricher = DataEnricher(uw, polygon, eodhd)
scorer = V31ScoringEngine()
ladder = LadderDetector(uw)
queue = QueueManager(pending_path, inverse_path, validated_path)

tabs = st.tabs(["Scan", "Manual Score", "Validate T+1", "Queues", "Connections"])

# -------------------- Scan tab --------------------
with tabs[0]:
    st.subheader("Run Scanner")

    colA, colB = st.columns([1, 2], gap="large")
    with colA:
        run = st.button("Run scan", type="primary", use_container_width=True)
        st.write("")
        st.caption("Filter logic matches package: exclude indices; require Vol > OI; premium threshold applied on UW-side.")

    with colB:
        st.markdown(
            """
            **What this does**
            - Pulls UW flow alerts (limit)
            - Computes total premium from ask/bid/mid/no-side premiums and ask% (as in package)
            - Filters by min premium + Vol > OI + excluded tickers
            - Enriches with Polygon spot-at-time + S/R + wick
            - Optionally fetches IV history (EODHD; usually falls back to chain snapshot)
            - Detects ladder (simple multi-strike same expiry/type check)
            - Scores with v3.1 scoring engine and writes queues
            """
        )

    if run:
        if not (uw_token and polygon_key and eodhd_key):
            st.error("Please provide UW + Polygon + EODHD keys (sidebar or env vars).")
        else:
            with st.spinner("Fetching UW flows..."):
                raw = uw.get_flows(limit=limit)

            if not raw:
                st.warning("No UW flows returned (check token / endpoint).")
            else:
                excluded = EXCLUDED_TICKERS_DEFAULT if exclude_indices else set()
                results: List[Dict[str, Any]] = []
                skipped = 0

                with st.spinner("Filtering, enriching, scoring..."):
                    for f in raw:
                        # Compute premium + ask% exactly like the package logic
                        total_prem = (
                            safe_float(f.get("total_ask_side_prem", 0)) +
                            safe_float(f.get("total_bid_side_prem", 0)) +
                            safe_float(f.get("total_mid_side_prem", 0)) +
                            safe_float(f.get("total_no_side_prem", 0))
                        )
                        if total_prem < float(min_premium):
                            skipped += 1
                            continue

                        ticker = str(f.get("ticker", "")).upper()
                        if ticker in excluded:
                            skipped += 1
                            continue

                        vol = safe_int(f.get("total_size", 0))
                        oi = max(safe_int(f.get("open_interest", 1)), 1)

                        if require_vol_gt_oi and (vol <= oi):
                            skipped += 1
                            continue

                        # Enrich
                        enriched = enricher.enrich_trade(f, fetch_iv=fetch_iv)

                        # Ladder detect
                        is_ladder, related = ladder.detect(
                            ticker, enriched["strike"], enriched["option_type"], enriched["expiry"]
                        )
                        if is_ladder:
                            enriched["ladder_role"] = "ladder"
                            enriched["related_strikes"] = related
                        else:
                            enriched["related_strikes"] = []

                        scored = scorer.score(enriched)
                        results.append(scored)

                        # Queue logic from package
                        if scored.get("predictive_score", 0) >= 5:
                            queue.add_pending(scored)
                        if scored.get("predictive_score", 0) <= -3:
                            queue.add_inverse(scored)

                st.success(f"Scored {len(results)} trades • Skipped {skipped}")

                # Display
                if results:
                    # compact table
                    table = []
                    for r in results:
                        table.append({
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
                            "IVRamp": bool(r.get("iv_ramping", False)),
                            "Score": r.get("predictive_score"),
                            "Max": r.get("max_score"),
                            "Verdict": r.get("verdict"),
                            "Tags": ", ".join(r.get("category_tags", [])),
                        })
                    st.dataframe(table, use_container_width=True, hide_index=True)

                    st.divider()
                    st.subheader("Details")
                    for r in results[:50]:
                        header = f"{r['ticker']} {str(r['option_type']).upper()} ${r['strike']} {r['expiry']} • Score {r['predictive_score']}/{r['max_score']} • {r['verdict']}"
                        with st.expander(header, expanded=False):
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.write("**Spot**", round(safe_float(r.get("spot", 0)), 2))
                                st.write("**Spot Source**", r.get("spot_source"))
                                st.write("**Entry (CT)**", r.get("entry_timestamp"))
                            with c2:
                                st.write("**Premium**", pretty_money(safe_float(r.get("total_premium", 0))))
                                st.write("**Prem %**", f"{safe_float(r.get('premium_pct', 0)):.2f}%")
                                st.write("**Ask %**", f"{safe_float(r.get('ask_pct', 0)):.1f}%")
                                st.write("**Vol/OI**", f"{safe_float(r.get('vol_oi_ratio', 0)):.2f}x")
                            with c3:
                                st.write("**DTE**", r.get("dte"))
                                st.write("**Wick triggered**", bool(r.get("wick_triggered", False)))
                                st.write("**Support/Resistance**", (round(safe_float(r.get("support", 0)), 2),
                                                                 round(safe_float(r.get("resistance", 0)), 2)))
                                st.write("**Ladder**", r.get("ladder_role", "isolated"))
                                if r.get("related_strikes"):
                                    st.write("**Related strikes**", r.get("related_strikes"))

                            st.write("**Factors**")
                            st.write("\n".join([f"• {x}" for x in r.get("score_factors", [])]) or "—")
                            st.write("**Penalties**")
                            st.write("\n".join([f"• {x}" for x in r.get("score_penalties", [])]) or "—")

                            st.write("**IV history (if any)**")
                            st.json(r.get("iv_history", {}))

                            st.write("**Raw UW (debug)**")
                            st.json(r.get("_raw", {}))

# -------------------- Manual Score tab --------------------
with tabs[1]:
    st.subheader("Manual Score (paste a single UW-like trade)")
    st.caption("Useful when you want to test the formula without calling APIs (or to tweak inputs).")

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

    col1, col2 = st.columns([1, 1])
    with col1:
        do_enrich = st.checkbox("Enrich using APIs (Polygon/EODHD/Earnings)", value=True)
        do_ladder = st.checkbox("Run ladder detection (UW ticker flow)", value=False)
        do_iv = st.checkbox("Fetch IV (EODHD)", value=fetch_iv)
    with col2:
        st.write("")

    if st.button("Score this trade", type="primary"):
        try:
            raw_flow = json.loads(raw_text)
            if not isinstance(raw_flow, dict):
                raise ValueError("JSON must be an object")

            if do_enrich:
                if not (uw_token and polygon_key and eodhd_key):
                    st.error("Provide keys in sidebar to enrich via APIs.")
                else:
                    enriched = enricher.enrich_trade(raw_flow, fetch_iv=do_iv)
                    if do_ladder:
                        is_l, rel = ladder.detect(
                            enriched["ticker"], enriched["strike"], enriched["option_type"], enriched["expiry"]
                        )
                        if is_l:
                            enriched["ladder_role"] = "ladder"
                            enriched["related_strikes"] = rel
                    scored = scorer.score(enriched)
                    st.success(f"Score: {scored['predictive_score']} / max {scored['max_score']} • {scored['verdict']}")
                    st.json(scored)
            else:
                # score using provided/derived fields only (user must supply required fields)
                scored = scorer.score(raw_flow)
                st.json(scored)
        except Exception as e:
            st.error(f"Error: {e}")

# -------------------- Validate T+1 tab --------------------
with tabs[2]:
    st.subheader("Validate T+1 (manual inputs)")
    st.caption(
        "The package menu notes T+1 validation as 'coming soon'. "
        "This UI lets you manually apply the T+1 confirmation logic you track (OI change + next-day IV)."
    )

    pending = queue.load_queue(pending_path)
    if not pending:
        st.info("No pending trades in queue yet. Run a scan first.")
    else:
        choices = [f"{i}: {t.get('ticker')} {str(t.get('option_type','')).upper()} {t.get('strike')} {t.get('expiry')} • score {t.get('predictive_score')}"
                   for i, t in enumerate(pending)]
        sel = st.selectbox("Select pending trade", choices, index=0)
        idx = int(sel.split(":")[0])
        trade = pending[idx]

        st.write("**Selected trade**")
        st.json({k: trade.get(k) for k in [
            "ticker", "option_type", "strike", "expiry", "entry_timestamp",
            "predictive_score", "max_score", "verdict", "premium_pct", "ask_pct",
            "volume", "open_interest", "iv_history", "iv_ramping", "clean_exception"
        ]})

        st.divider()
        st.write("**Enter T+1 confirmation data** (from your broker/IV panel/screenshots)")

        c1, c2, c3 = st.columns(3)
        with c1:
            prior_oi = st.number_input("Prior-day OI", min_value=0, value=int(trade.get("open_interest", 0)), step=1)
            t1_oi = st.number_input("T+1 OI", min_value=0, value=int(trade.get("open_interest", 0)), step=1)
        with c2:
            iv_entry = st.number_input("Entry-day IV (%)", min_value=0.0, value=0.0, step=0.5)
            iv_t1 = st.number_input("T+1 IV (%)", min_value=0.0, value=0.0, step=0.5)
        with c3:
            roll_override = st.checkbox("Roll override context (manual)", value=False,
                                        help="If you identified a roll, waive execution/isolation penalties and add +1 continuation.")

        if st.button("Apply T+1 validation + archive to validated queue", type="primary"):
            # Copy trade and apply validation adjustments aligned to ruleset notes:
            # - OI change must be explicit; if low conversion, flag trap risk
            # - Next-day IV up => +1, down => -1
            # - Roll override: present adjusted score (simple implementation)
            t = dict(trade)

            vol = safe_int(t.get("volume", 0))
            oi_change = max(int(t1_oi - prior_oi), 0)
            oi_conv_pct = (oi_change / max(vol, 1)) * 100.0

            # Store validation fields
            t["prior_oi_tminus1"] = int(prior_oi)
            t["tplus1_oi"] = int(t1_oi)
            t["oi_change_tplus1"] = int(oi_change)
            t["oi_change_pct_of_volume"] = round(oi_conv_pct, 2)
            t["iv_entry_manual"] = float(iv_entry)
            t["iv_tplus1_manual"] = float(iv_t1)

            # Validation delta
            validation_delta = 0

            # Next-day IV behavior (ruleset: up +1, down -1)
            if iv_entry > 0 and iv_t1 > 0:
                if iv_t1 > iv_entry:
                    validation_delta += 1
                    t.setdefault("validation_notes", []).append("T+1 IV up (+1)")
                elif iv_t1 < iv_entry:
                    validation_delta -= 1
                    t.setdefault("validation_notes", []).append("T+1 IV down (-1)")

            # OI conversion trap flag guidance (ruleset warns mismatch)
            if oi_conv_pct < 10.0:
                t.setdefault("validation_notes", []).append("OI conversion <10% of volume (trap flag)")
            elif oi_conv_pct >= 50.0:
                t.setdefault("validation_notes", []).append("Strong OI conversion (>=50% of volume)")
            else:
                t.setdefault("validation_notes", []).append("Moderate OI conversion (10–50% of volume)")

            # Roll override: lightweight adjustment consistent with ruleset description
            # (We do not have full execution-side breakdown types here, so we only add a continuation bonus.)
            if roll_override:
                validation_delta += 1
                t.setdefault("validation_notes", []).append("Roll override continuation (+1)")

            pred = safe_int(t.get("predictive_score", 0))
            t["validated_score"] = pred + validation_delta

            # simple validated verdict
            vs = t["validated_score"]
            if vs >= 8:
                t["validated_verdict"] = "HIGH CONVICTION"
            elif vs >= 7:
                t["validated_verdict"] = "TRADEABLE"
            elif vs >= 6:
                t["validated_verdict"] = "MODERATE"
            else:
                t["validated_verdict"] = "TRAP / SKIP"

            # archive
            queue.add_validated(t)

            # remove from pending
            pending.pop(idx)
            queue.save_queue(pending_path, pending)

            st.success(f"Archived to validated queue. Validated score: {t['validated_score']} • {t['validated_verdict']}")
            st.json(t)

# -------------------- Queues tab --------------------
with tabs[3]:
    st.subheader("Queues")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("### Pending")
        pend = queue.load_queue(pending_path)
        st.write(f"{len(pend)} items")
        st.dataframe([
            {"Ticker": t.get("ticker"), "Strike": t.get("strike"), "Type": t.get("option_type"),
             "Expiry": t.get("expiry"), "Score": t.get("predictive_score"), "Verdict": t.get("verdict")}
            for t in pend[-200:]
        ], use_container_width=True, hide_index=True)

    with c2:
        st.write("### Inverse")
        inv = queue.load_queue(inverse_path)
        st.write(f"{len(inv)} items")
        st.dataframe([
            {"Ticker": t.get("ticker"), "Strike": t.get("strike"), "Type": t.get("option_type"),
             "Expiry": t.get("expiry"), "Score": t.get("predictive_score"), "Verdict": t.get("verdict")}
            for t in inv[-200:]
        ], use_container_width=True, hide_index=True)

    with c3:
        st.write("### Validated")
        val = queue.load_queue(validated_path)
        st.write(f"{len(val)} items")
        st.dataframe([
            {"Ticker": t.get("ticker"), "Strike": t.get("strike"), "Type": t.get("option_type"),
             "Expiry": t.get("expiry"), "PredScore": t.get("predictive_score"),
             "ValScore": t.get("validated_score"), "ValVerdict": t.get("validated_verdict")}
            for t in val[-200:]
        ], use_container_width=True, hide_index=True)

    st.divider()
    if st.button("Reset all queues (danger)", type="secondary"):
        queue.save_queue(pending_path, [])
        queue.save_queue(inverse_path, [])
        queue.save_queue(validated_path, [])
        st.warning("Queues cleared.")

# -------------------- Connections tab --------------------
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
        **Reminder:** The package documents EODHD historical IV endpoint returning 422, which blocks true IV-ramp history.
        This app follows the same fallback behavior (chain snapshot only), so most scans will show `IVRamp=False`
        unless you provide multi-day IV via a different source.
        """
    )
