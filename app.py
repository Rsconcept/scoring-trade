#!/usr/bin/env python3
"""
Streamlit v3.9 Options Flow Scanner — Contract IV + Underlying IV FIXED (always builds)
=====================================================================================
What’s fixed vs v3.8:
✅ Underlying IV now builds even when your scan filters out ATM-ish trades (ex: min premium = $1M).
   - If scan doesn't find ATM-ish contract IV for a ticker, we auto-fetch UW ticker flow for that ticker
     and pick the closest-to-ATM contract IV (using UW underlying_price + strike) and use its IV as
     Underlying ATM IV proxy.
✅ Then Underlying IV is assigned to ALL rows for that ticker, and stored daily to iv_history_store.json.

Notes:
- Contract IV comes from UW iv_start/iv_end (fraction -> percent).
- Underlying ATM IV is a PROXY built from the best (closest-to-ATM) contract IV we can find.
- Market closed:
  - UW still returns iv_start/iv_end for alerts/flows (often yes), so Contract IV can show.
  - Underlying IV can show if we can find a near-ATM flow in UW ticker flow OR if it exists in local store.

requirements.txt:
streamlit
requests

Secrets:
UW_TOKEN, EODHD_API_KEY
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


def today_yyyy_mm_dd() -> str:
    return date.today().strftime("%Y-%m-%d")


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


def calculate_dte(expiry_yyyy_mm_dd: str) -> int:
    try:
        exp_date = datetime.strptime(expiry_yyyy_mm_dd, "%Y-%m-%d")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return (exp_date - today).days
    except Exception:
        return 0


def normalize_iv_pct(iv: float) -> float:
    """Accepts fraction (0.59) or percent (59). Returns percent."""
    if iv <= 0:
        return 0.0
    return iv * 100.0 if iv < 1.0 else iv


def uw_epoch_ms_to_ct_string(epoch_ms: Any) -> str:
    try:
        ms = int(epoch_ms)
        dt_utc = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        dt_ct = dt_utc + timedelta(hours=CT_OFFSET)
        return dt_ct.strftime("%Y-%m-%d %I:%M:%S %p CT")
    except Exception:
        return ""


def to_ct_string(any_time: Any) -> str:
    if any_time is None:
        return ""
    if isinstance(any_time, (int, float)) and any_time > 1_000_000_000:
        return uw_epoch_ms_to_ct_string(any_time)
    if isinstance(any_time, str) and any_time.isdigit():
        return uw_epoch_ms_to_ct_string(any_time)
    try:
        s = str(any_time)
        if "T" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ct = dt.astimezone(timezone.utc) + timedelta(hours=CT_OFFSET)
            return ct.strftime("%Y-%m-%d %I:%M:%S %p CT")
        return s
    except Exception:
        return str(any_time)


def contract_key(ticker: str, expiry: str, option_type: str, strike: float) -> str:
    return f"{ticker.upper()}|{expiry}|{option_type.lower()}|{float(strike):.2f}"


def underlying_iv_key(ticker: str) -> str:
    return f"{ticker.upper()}|UNDERLYING_ATM_IV"


def extract_uw_iv(raw_flow: Dict[str, Any]) -> float:
    """Handles UW iv_start / iv_end (fraction -> percent)."""
    if not isinstance(raw_flow, dict):
        return 0.0

    ivs = normalize_iv_pct(safe_float(raw_flow.get("iv_start", 0)))
    ive = normalize_iv_pct(safe_float(raw_flow.get("iv_end", 0)))
    if ivs > 0:
        return float(ivs)
    if ive > 0:
        return float(ive)

    for k in ("iv", "impliedVolatility", "implied_volatility", "implied_vol"):
        v = normalize_iv_pct(safe_float(raw_flow.get(k, 0)))
        if v > 0:
            return float(v)

    g = raw_flow.get("greeks")
    if isinstance(g, dict):
        for k in ("iv", "impliedVolatility", "implied_volatility"):
            v = normalize_iv_pct(safe_float(g.get(k, 0)))
            if v > 0:
                return float(v)

    return 0.0


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
        iv_value = float(iv_value)
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
                r["iv"] = iv_value
                replaced = True
                break
        if not replaced:
            rows.append({"date": t, "iv": iv_value})

        rows = [r for r in rows if isinstance(r, dict) and r.get("date") and r.get("iv") is not None]
        rows.sort(key=lambda r: r["date"])
        rows = rows[-180:]

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

    def get_flows(self, limit: int = 200) -> List[Dict[str, Any]]:
        status, data, _ = http_get(
            f"{self.BASE_URL}/option-trades/flow-alerts",
            headers=self.headers,
            params={"limit": limit},
        )
        if status != 200 or not data:
            return []
        return data.get("data", []) or []

    def get_ticker_flow(self, ticker: str, limit: int = 250) -> List[Dict[str, Any]]:
        ticker = ticker.strip().upper()
        status, data, _ = http_get(
            f"{self.BASE_URL}/stock/{ticker}/options-flow",
            headers=self.headers,
            params={"limit": limit},
        )
        if status == 200 and data:
            return data.get("data", []) or []
        return []


class EODHDAPI:
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

    def get_spot_realtime(self, ticker: str) -> Tuple[float, str]:
        ticker = ticker.strip().upper()
        if not self.api_key:
            return 0.0, "EODHD spot: missing api key"
        url = f"{self.BASE_URL}/real-time/{ticker}.US"
        status, data, err = http_get(url, params={"api_token": self.api_key, "fmt": "json"}, timeout=20)
        if status == 200 and isinstance(data, dict):
            for k in ("close", "price", "last", "last_price"):
                v = safe_float(data.get(k, 0))
                if v > 0:
                    return v, f"EODHD real-time ({k})"
        return 0.0, f"EODHD real-time error: {err or 'no price keys'}"


# -------------------- Enricher --------------------

class DataEnricher:
    def __init__(self, eodhd: EODHDAPI, iv_store: LocalIVStore, iv_lookback_days: int, iv_require_strict: bool):
        self.eodhd = eodhd
        self.iv_store = iv_store
        self.iv_lookback_days = iv_lookback_days
        self.iv_require_strict = iv_require_strict

    def enrich_trade(self, raw_flow: Dict[str, Any], use_contract_iv: bool) -> Dict[str, Any]:
        ticker = str(raw_flow.get("ticker", "")).upper().strip()

        option_type = str(raw_flow.get("option_type") or raw_flow.get("type") or "call").lower().strip()
        if option_type in ("calls", "call"):
            option_type = "call"
        if option_type in ("puts", "put"):
            option_type = "put"

        strike = safe_float(raw_flow.get("strike", 0))
        expiry = str(raw_flow.get("expiry", "")).strip()
        entry_timestamp = to_ct_string(raw_flow.get("start_time") or raw_flow.get("created_at") or "")

        # Spot: prefer UW underlying_price, fallback to EODHD real-time
        spot = safe_float(raw_flow.get("underlying_price", 0))
        spot_source = "UW underlying_price"
        if spot <= 0:
            spot, spot_source = self.eodhd.get_spot_realtime(ticker)

        strike_dist_pct = (abs(strike - spot) / spot * 100.0) if spot > 0 else 999.0
        dte = calculate_dte(expiry)

        total_prem = (
            safe_float(raw_flow.get("total_ask_side_prem", 0))
            + safe_float(raw_flow.get("total_bid_side_prem", 0))
            + safe_float(raw_flow.get("total_mid_side_prem", 0))
            + safe_float(raw_flow.get("total_no_side_prem", 0))
        )
        if total_prem <= 0:
            total_prem = safe_float(raw_flow.get("total_premium", 0))

        ask_prem = safe_float(raw_flow.get("total_ask_side_prem", 0))
        ask_pct = (ask_prem / total_prem * 100.0) if total_prem > 0 else 0.0

        volume = safe_int(raw_flow.get("total_size", raw_flow.get("volume", 0)))
        oi = safe_int(raw_flow.get("open_interest", 0))
        vol_oi_ratio = (volume / oi) if oi > 0 else 999.0

        denom = (spot * 100.0 * max(volume, 1)) if spot > 0 else 0.0
        premium_pct = (total_prem / denom * 100.0) if denom > 0 else 0.0

        ckey = contract_key(ticker, expiry, option_type, strike)

        current_iv = 0.0
        contract_iv_reason = "disabled"
        if use_contract_iv:
            iv_uw = extract_uw_iv(raw_flow)
            if iv_uw > 0:
                current_iv = iv_uw
                contract_iv_reason = "UW iv_start/iv_end"
                self.iv_store.upsert_today(ckey, current_iv)
            else:
                contract_iv_reason = "UW missing IV fields"

        iv_ramping, ramp_points = self.iv_store.detect_ramp(
            ckey, lookback_days=self.iv_lookback_days, require_strict=self.iv_require_strict
        )

        return {
            "ticker": ticker,
            "option_type": option_type,
            "strike": strike,
            "expiry": expiry,
            "entry_timestamp": entry_timestamp,

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

            "contract_key": ckey,
            "current_iv": float(current_iv),
            "contract_iv_reason": contract_iv_reason,
            "iv_ramping": bool(iv_ramping),
            "iv_ramp_points": ramp_points,

            "iv_underlying_current": 0.0,
            "iv_underlying_source": "Not computed yet",
            "iv_underlying_points_7d": [],
            "iv_underlying_7d_change": 0.0,

            "has_sweep": bool(raw_flow.get("has_sweep", raw_flow.get("is_sweep", False))),
            "category_tags": [],
            "_raw": raw_flow,
        }


# -------------------- Scoring --------------------

class V39ScoringEngine:
    def score(self, r: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        factors: List[str] = []
        penalties: List[str] = []
        tags: List[str] = []

        prem_pct = safe_float(r.get("premium_pct", 0))
        if 2.5 <= prem_pct <= 5.0:
            score += 2; factors.append(f"Premium {prem_pct:.1f}% (+2)")
        elif 1.0 <= prem_pct < 2.5:
            score += 1; factors.append(f"Premium {prem_pct:.1f}% (+1)")
        elif prem_pct < 1.0 and safe_float(r.get("spot", 0)) > 0:
            score -= 2; penalties.append(f"Ultra-low premium {prem_pct:.2f}% (-2)")

        dist = safe_float(r.get("strike_dist_pct", 999))
        if safe_float(r.get("spot", 0)) > 0:
            if dist <= 7:
                score += 2; factors.append(f"Strike {dist:.1f}% OTM (+2)")
            elif dist > 15:
                score -= 2; penalties.append(f"Strike {dist:.1f}% deep OTM (-2)")

        dte = safe_int(r.get("dte", 0))
        if 7 <= dte <= 21:
            score += 1; factors.append(f"DTE {dte}d (+1)")
        elif dte <= 1:
            score -= 1; penalties.append("0-1 DTE (-1)")

        ask_pct = safe_float(r.get("ask_pct", 0))
        if ask_pct >= 70:
            score += 1; factors.append(f"Ask {ask_pct:.0f}% (+1)")
        elif 0 < ask_pct < 30:
            score -= 2; penalties.append(f"Bid/mid heavy (Ask {ask_pct:.0f}%) (-2)")

        vol_oi = safe_float(r.get("vol_oi_ratio", 0))
        if vol_oi >= 2:
            score += 2; factors.append(f"Vol/OI {vol_oi:.1f}x (+2)")
        elif vol_oi >= 1:
            score += 1; factors.append(f"Vol/OI {vol_oi:.1f}x (+1)")

        ivc = safe_float(r.get("current_iv", 0))
        ivu = safe_float(r.get("iv_underlying_current", 0))

        if ivc > 0:
            tags.append("HasContractIV")
            if bool(r.get("iv_ramping", False)):
                score += 1; factors.append(f"Contract IV ramp (+1) {r.get('iv_ramp_points')}")
        else:
            tags.append("NoContractIV")

        if ivu > 0:
            tags.append("HasUnderlyingIV")
            iv7 = safe_float(r.get("iv_underlying_7d_change", 0))
            if iv7 >= 3:
                score += 1; factors.append(f"Underlying IV +{iv7:.1f} in 7d (+1)")
        else:
            tags.append("NoUnderlyingIV")

        max_score = 12 if (ivc > 0 or ivu > 0) else 6
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

        r["predictive_score"] = int(final_score)
        r["max_score"] = int(max_score)
        r["score_factors"] = factors
        r["score_penalties"] = penalties
        r["verdict"] = verdict
        r["category_tags"] = list(dict.fromkeys(tags))
        return r


# -------------------- Queue Manager --------------------

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


# -------------------- Underlying IV Builder (NEW) --------------------

@st.cache_data(ttl=120, show_spinner=False)
def cached_ticker_flow(token: str, ticker: str, limit: int) -> List[Dict[str, Any]]:
    uw = UnusualWhalesAPI(token)
    return uw.get_ticker_flow(ticker, limit=limit)


def find_best_atm_iv_from_flows(
    flows: List[Dict[str, Any]],
    max_atm_dist_pct: float,
) -> Optional[Dict[str, Any]]:
    """
    Returns {"iv":..., "dist":..., "spot":..., "strike":..., "source":...}
    Picks the closest-to-ATM contract IV from the provided flow list.
    """
    best: Optional[Dict[str, Any]] = None
    for f in flows:
        try:
            spot = safe_float(f.get("underlying_price", 0))
            strike = safe_float(f.get("strike", 0))
            if spot <= 0 or strike <= 0:
                continue
            dist = abs(strike - spot) / spot * 100.0
            if dist > max_atm_dist_pct:
                continue

            iv = extract_uw_iv(f)
            if iv <= 0:
                continue

            if best is None or dist < safe_float(best.get("dist", 999)):
                best = {
                    "iv": float(iv),
                    "dist": float(dist),
                    "spot": float(spot),
                    "strike": float(strike),
                    "source": "UW ticker_flow ATM proxy",
                }
        except Exception:
            continue
    return best


# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="v3.9 Options Flow Scanner (Underlying IV Fixed)", layout="wide")
st.title("v3.9 Options Flow Scanner (Contract IV + Underlying IV always builds)")
st.caption("Underlying IV now builds from UW ticker flow if scan doesn’t include ATM-ish trades.")

snapshot = SnapshotManager(SNAPSHOT_FILE)
iv_store = LocalIVStore(IV_STORE_FILE)

with st.sidebar:
    st.header("Data Source")
    data_mode = st.toggle("Replay Mode (ON = use saved snapshot)", value=False)
    meta = snapshot.get_meta()
    if meta.get("saved_at_utc"):
        st.write(f"Snapshot saved: `{meta.get('saved_at_utc')}`")
        st.write(f"Snapshot count: `{meta.get('count')}`")

    st.divider()
    st.header("API Keys")
    uw_token = st.text_input("Unusual Whales Token", value=os.getenv("UW_TOKEN", ""), type="password")
    eodhd_key = st.text_input("EODHD API Key", value=os.getenv("EODHD_API_KEY", ""), type="password")

    st.divider()
    st.subheader("Scan Controls")
    manual_ticker = st.text_input("Manual ticker scan (optional)", value="").strip().upper()
    st.caption("Live mode: if set, uses UW ticker flow for that ticker. Replay: filters snapshot.")

    use_contract_iv = st.checkbox("Use Contract IV (UW iv_start/iv_end) + store", value=True)

    limit = st.slider("UW flow alerts limit (Live only)", 10, 250, 200, 10)

    # ✅ defaults requested
    min_premium = st.number_input("Min premium ($)", min_value=0, value=1_000_000, step=25_000)
    min_size = st.number_input("Min size (contracts)", min_value=0, value=0, step=100)
    min_vol_oi = st.number_input("Min Vol/OI", min_value=0.0, value=1.0, step=0.1)

    require_vol_gt_oi = st.checkbox("Require Vol > OI", value=True)
    exclude_indices = st.checkbox("Exclude indices + major ETFs", value=True)

    st.divider()
    st.subheader("Underlying IV Settings")
    max_atm_dist_pct = st.slider("Max distance to consider ATM-ish (%)", 0.5, 20.0, 7.0, 0.5)
    ticker_flow_fallback = st.checkbox("Fallback: build Underlying IV from UW ticker flow", value=True)
    ticker_flow_limit = st.slider("Ticker-flow depth (fallback)", 50, 500, 250, 50)

    st.divider()
    st.subheader("IV Store")
    st.download_button(
        "Download iv_history_store.json",
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
eodhd = EODHDAPI(eodhd_key)

enricher = DataEnricher(
    eodhd=eodhd,
    iv_store=iv_store,
    iv_lookback_days=3,
    iv_require_strict=True,
)
scorer = V39ScoringEngine()
queue = QueueManager(PENDING_TRADES_FILE, INVERSE_SIGNALS_FILE, VALIDATED_TRADES_FILE)

tabs = st.tabs(["Scan", "Connections"])


def get_source_flows() -> Tuple[List[Dict[str, Any]], str]:
    if data_mode:
        flows = snapshot.load()
        if manual_ticker:
            flows = [f for f in flows if str(f.get("ticker", "")).upper().strip() == manual_ticker]
            return flows, f"Replay (snapshot filtered: {manual_ticker})"
        return flows, "Replay (snapshot)"

    if manual_ticker:
        return uw.get_ticker_flow(manual_ticker, limit=min(600, max(250, limit))), f"Live (UW ticker flow: {manual_ticker})"
    return uw.get_flows(limit=limit), "Live (UW flow alerts)"


# -------------------- Scan --------------------

with tabs[0]:
    st.subheader("Run Scanner")
    run = st.button("Run scan", type="primary", use_container_width=True)

    if run:
        if not data_mode and not uw_token:
            st.error("Live Mode requires UW_TOKEN.")
        else:
            skip_reasons = {
                "bad_ticker": 0, "premium": 0, "excluded": 0,
                "min_size": 0, "vol_oi": 0, "min_vol_oi": 0
            }

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

                # best ATM-ish contract IV per ticker found in the *included* scan rows
                best_underlying_iv: Dict[str, Dict[str, Any]] = {}

                with st.spinner("Filtering, enriching, scoring..."):
                    for f in flows:
                        ticker = str(f.get("ticker", "")).upper().strip()
                        if not ticker or len(ticker) > 8:
                            skip_reasons["bad_ticker"] += 1
                            continue

                        total_prem = (
                            safe_float(f.get("total_ask_side_prem", 0))
                            + safe_float(f.get("total_bid_side_prem", 0))
                            + safe_float(f.get("total_mid_side_prem", 0))
                            + safe_float(f.get("total_no_side_prem", 0))
                        )
                        if total_prem <= 0:
                            total_prem = safe_float(f.get("total_premium", 0))

                        if total_prem < float(min_premium):
                            skip_reasons["premium"] += 1
                            continue

                        if ticker in excluded:
                            skip_reasons["excluded"] += 1
                            continue

                        vol = safe_int(f.get("total_size", f.get("volume", 0)))
                        oi = safe_int(f.get("open_interest", 0))

                        if vol < int(min_size):
                            skip_reasons["min_size"] += 1
                            continue

                        if require_vol_gt_oi and (oi > 0) and (vol <= oi):
                            skip_reasons["vol_oi"] += 1
                            continue

                        vol_oi_ratio = (vol / oi) if oi > 0 else 999.0
                        if vol_oi_ratio < float(min_vol_oi):
                            skip_reasons["min_vol_oi"] += 1
                            continue

                        enriched = enricher.enrich_trade(f, use_contract_iv=bool(use_contract_iv))

                        # If this row is ATM-ish and has contract IV, it can become underlying IV proxy
                        ivc = safe_float(enriched.get("current_iv", 0))
                        dist = safe_float(enriched.get("strike_dist_pct", 999))
                        if ivc > 0 and dist <= float(max_atm_dist_pct):
                            cur = best_underlying_iv.get(ticker)
                            if (cur is None) or (dist < safe_float(cur.get("dist", 999))):
                                best_underlying_iv[ticker] = {"iv": ivc, "dist": dist, "source": "Scan ATM proxy"}

                        scored = scorer.score(enriched)
                        results.append(scored)

                        if safe_int(scored.get("predictive_score", 0)) >= 5:
                            queue.add_pending(scored)
                        if safe_int(scored.get("predictive_score", 0)) <= -3:
                            queue.add_inverse(scored)

                # ✅ NEW: For tickers in results with no ATM proxy, fetch ticker_flow and build it anyway
                tickers_in_results = sorted({str(r.get("ticker", "")).upper().strip() for r in results if r.get("ticker")})
                if ticker_flow_fallback and uw_token and tickers_in_results:
                    with st.spinner("Building Underlying IV proxies from UW ticker flow (fallback)..."):
                        for tkr in tickers_in_results:
                            if tkr in best_underlying_iv:
                                continue
                            tf = cached_ticker_flow(uw_token, tkr, int(ticker_flow_limit))
                            best = find_best_atm_iv_from_flows(tf, float(max_atm_dist_pct))
                            if best and safe_float(best.get("iv", 0)) > 0:
                                best_underlying_iv[tkr] = {"iv": best["iv"], "dist": best["dist"], "source": best["source"]}

                # store best underlying IV per ticker for today
                for tkr, info in best_underlying_iv.items():
                    iv_store.upsert_today(underlying_iv_key(tkr), float(info["iv"]))

                # assign underlying IV to ALL rows (best from scan/tickerflow, else last stored)
                for r in results:
                    tkr = str(r.get("ticker", "")).upper().strip()
                    ukey = underlying_iv_key(tkr)

                    best = best_underlying_iv.get(tkr)
                    if best and safe_float(best.get("iv", 0)) > 0:
                        r["iv_underlying_current"] = float(best["iv"])
                        r["iv_underlying_source"] = f"{best.get('source','ATM proxy')} (dist {best.get('dist',0):.1f}%)"
                    else:
                        pts = iv_store.last_n_points(ukey, 7)
                        if pts:
                            r["iv_underlying_current"] = float(pts[-1][1])
                            r["iv_underlying_source"] = "Local store (most recent)"
                        else:
                            r["iv_underlying_current"] = 0.0
                            r["iv_underlying_source"] = "No underlying IV yet (run daily to build)"

                    r["iv_underlying_points_7d"] = iv_store.last_n_points(ukey, 7)
                    pts = r["iv_underlying_points_7d"]
                    r["iv_underlying_7d_change"] = float(pts[-1][1] - pts[0][1]) if len(pts) >= 2 else 0.0

                # re-score after underlying IV assignment
                results = [scorer.score(r) for r in results]

                st.success(f"Source: {src} • Scored {len(results)} trades")
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
                                "SpotSrc": str(r.get("spot_source", ""))[:28],
                                "Premium$": round(safe_float(r.get("total_premium", 0.0))),
                                "Ask%": round(safe_float(r.get("ask_pct", 0.0)), 1),
                                "Vol/OI": round(safe_float(r.get("vol_oi_ratio", 0.0)), 2),
                                "IV(Contract)%": round(safe_float(r.get("current_iv", 0.0)), 2),
                                "IVcSrc": str(r.get("contract_iv_reason", ""))[:28],
                                "IV(Underlying)%": round(safe_float(r.get("iv_underlying_current", 0.0)), 2),
                                "IVuSrc": str(r.get("iv_underlying_source", ""))[:34],
                                "IV7dΔ": round(safe_float(r.get("iv_underlying_7d_change", 0.0)), 2),
                                "Score": r.get("predictive_score"),
                                "Max": r.get("max_score"),
                                "Verdict": r.get("verdict"),
                                "Tags": ", ".join(r.get("category_tags", [])),
                            }
                        )
                    st.dataframe(table, use_container_width=True, hide_index=True)

                    st.divider()
                    st.subheader("Details (Top 20)")
                    for r in results[:20]:
                        header = (
                            f"{r['ticker']} {str(r['option_type']).upper()} ${r['strike']} {r['expiry']} • "
                            f"IVc {safe_float(r.get('current_iv',0)):.2f}% • "
                            f"IVu {safe_float(r.get('iv_underlying_current',0)):.2f}% • "
                            f"Score {r['predictive_score']}"
                        )
                        with st.expander(header, expanded=False):
                            st.write("**Entry (CT):**", r.get("entry_timestamp"))
                            st.write("**Spot Source:**", r.get("spot_source"))
                            st.write("**Contract IV reason:**", r.get("contract_iv_reason"))
                            st.write("**Underlying IV source:**", r.get("iv_underlying_source"))
                            st.write("**Underlying IV 7D points:**")
                            st.json(r.get("iv_underlying_points_7d", []))
                            st.write("**Raw UW (debug):**")
                            st.json(r.get("_raw", {}))


# -------------------- Connections --------------------

with tabs[1]:
    st.subheader("Test Connections")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Test UW"):
            ok, msg = uw.test_connection()
            st.success(msg) if ok else st.error(msg)

    with c2:
        if st.button("Test EODHD"):
            ok, msg = eodhd.test_connection()
            st.success(msg) if ok else st.error(msg)
