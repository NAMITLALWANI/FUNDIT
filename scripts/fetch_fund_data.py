"""
Snapshot public mutual fund data into data/raw/ so the project is reproducible offline.

Sources (documented in docs/data.md):
  * AMFI daily NAV history, accessed through the public mirror https://api.mfapi.in
    -> data/raw/nav/<scheme_code>.json
  * Kuvera public scheme attribute API (expense ratio, AUM, SEBI riskometer, objective,
    minimum SIP / lump sum, lock-in) https://api.kuvera.in/mf/api/v5/fund_schemes/<code>.json
    -> data/raw/scheme_attributes/<scheme_code>.json

Usage:
    python scripts/fetch_fund_data.py            # fetch every fund in data/raw/fund_universe.json
    python scripts/fetch_fund_data.py --only 122639 118825

The script is idempotent: re-running overwrites the snapshot and records the collection date in
data/raw/sources.json.
"""

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
UNIVERSE_PATH = RAW_DIR / "fund_universe.json"
NAV_DIR = RAW_DIR / "nav"
ATTR_DIR = RAW_DIR / "scheme_attributes"
SOURCES_PATH = RAW_DIR / "sources.json"

MFAPI_URL = "https://api.mfapi.in/mf/{code}"
KUVERA_URL = "https://api.kuvera.in/mf/api/v5/fund_schemes/{code}.json"
KUVERA_DROP_FIELDS = {"comparison", "sip_dates", "upsizecode_sip_dates", "sips"}
HEADERS = {"User-Agent": "ai-decision-engine-v2/0.2 (public data snapshot; contact via repository)"}


def fetch_json(client: httpx.Client, url: str, retries: int = 3) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = client.get(url, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            time.sleep(0.5 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def snapshot_nav(client: httpx.Client, scheme_code: int) -> int:
    payload = fetch_json(client, MFAPI_URL.format(code=scheme_code))
    if payload.get("status") != "SUCCESS" or not payload.get("data"):
        raise RuntimeError(f"mfapi returned no NAV data for {scheme_code}")
    payload["_source"] = {
        "provider": "AMFI via api.mfapi.in",
        "url": MFAPI_URL.format(code=scheme_code),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    (NAV_DIR / f"{scheme_code}.json").write_text(json.dumps(payload), encoding="utf-8")
    return len(payload["data"])


def snapshot_attributes(client: httpx.Client, scheme_code: int, kuvera_code: str, isin: Optional[str]) -> bool:
    payload = fetch_json(client, KUVERA_URL.format(code=kuvera_code))
    if not payload or not isinstance(payload, list):
        return False
    record: Dict[str, Any] = {k: v for k, v in payload[0].items() if k not in KUVERA_DROP_FIELDS}
    if isin and record.get("ISIN") and record["ISIN"] != isin:
        raise RuntimeError(f"ISIN mismatch for {scheme_code}: {record.get('ISIN')} != {isin}")
    record["_source"] = {
        "provider": "Kuvera public scheme API",
        "url": KUVERA_URL.format(code=kuvera_code),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    (ATTR_DIR / f"{scheme_code}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return True


def write_sources(fund_count: int, nav_points: int, attr_count: int) -> None:
    today = date.today().isoformat()
    sources = {
        "collected_at": today,
        "funds_in_universe": fund_count,
        "nav_points": nav_points,
        "attribute_records": attr_count,
        "sources": [
            {
                "source_id": "amfi_mfapi",
                "name": "AMFI India daily NAV history (via api.mfapi.in mirror)",
                "url": "https://api.mfapi.in",
                "collected_at": today,
                "description": (
                    "Daily NAV series per scheme as published by the Association of Mutual Funds in India "
                    "(AMFI). Used to compute returns, volatility, drawdown and Sharpe ratio."
                ),
                "license_notes": (
                    "AMFI NAV data is public information. api.mfapi.in is a free, unofficial JSON mirror; "
                    "values should be cross-checked against amfiindia.com for production use."
                ),
            },
            {
                "source_id": "kuvera_scheme_api",
                "name": "Kuvera public scheme attribute API",
                "url": "https://api.kuvera.in/mf/api/v5/fund_schemes/",
                "collected_at": today,
                "description": (
                    "Scheme-level attributes aggregated from AMC disclosures: total expense ratio (with date), "
                    "AUM, SEBI riskometer label, investment objective, minimum SIP/lump sum, lock-in, fund manager."
                ),
                "license_notes": (
                    "Unofficial public endpoint of a SEBI-registered platform; no API key required. Attributes "
                    "originate from AMC factsheets/SIDs and should be verified against the AMC document page "
                    "recorded in each fund's source_url before any real investment decision."
                ),
            },
        ],
    }
    SOURCES_PATH.write_text(json.dumps(sources, indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", type=int, help="Restrict to these AMFI scheme codes")
    parser.add_argument("--skip-attributes", action="store_true", help="Only refresh NAV history")
    args = parser.parse_args(argv)

    universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))["funds"]
    if args.only:
        universe = [f for f in universe if f["amfi_scheme_code"] in set(args.only)]

    NAV_DIR.mkdir(parents=True, exist_ok=True)
    ATTR_DIR.mkdir(parents=True, exist_ok=True)

    total_points = 0
    attr_count = 0
    failures: List[str] = []
    with httpx.Client(headers=HEADERS) as client:
        for fund in universe:
            code = fund["amfi_scheme_code"]
            try:
                points = snapshot_nav(client, code)
                total_points += points
                attr_status = "skipped"
                if not args.skip_attributes and fund.get("kuvera_code"):
                    if snapshot_attributes(client, code, fund["kuvera_code"], fund.get("isin")):
                        attr_count += 1
                        attr_status = "ok"
                    else:
                        attr_status = "missing"
                elif not fund.get("kuvera_code"):
                    attr_status = "no-kuvera-code"
                print(f"[ok] {code} {fund['scheme_name'][:60]:60s} nav={points:5d} attributes={attr_status}")
            except Exception as exc:  # noqa: BLE001 - report every failure, continue with the rest
                failures.append(f"{code}: {exc}")
                print(f"[fail] {code} {exc}")
            time.sleep(0.3)

    write_sources(len(universe), total_points, attr_count)
    print(f"\nSnapshot complete: {len(universe) - len(failures)}/{len(universe)} funds, {total_points} NAV points, {attr_count} attribute records")
    if failures:
        print("Failures:")
        for f in failures:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
