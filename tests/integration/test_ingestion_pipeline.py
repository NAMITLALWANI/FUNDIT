"""
Ingestion pipeline integration test on a SYNTHETIC raw snapshot written to a temp directory.

The snapshot mimics the mfapi/Kuvera file layout with made-up values so the test is hermetic.
It never touches data/raw and does not represent any real scheme.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from app.db.repository import FundFilter, FundRepository
from app.db.session import create_db_engine, get_session_factory, init_database
from app.ingestion.metrics import compute_fund_metrics
from app.ingestion.models import NavPoint
from app.ingestion.pipeline import IngestionPipeline


def _write_snapshot(raw: Path, docs: Path) -> None:
    (raw / "nav").mkdir(parents=True)
    (raw / "scheme_attributes").mkdir()
    (docs / "regulatory").mkdir(parents=True)

    (raw / "fund_universe.json").write_text(json.dumps({"funds": [
        {"fund_id": "SYN-1", "amfi_scheme_code": 1, "scheme_name": "Synthetic One Fund - Direct Plan - Growth", "kuvera_code": "X1"},
        {"fund_id": "SYN-2", "amfi_scheme_code": 2, "scheme_name": "Synthetic Two Fund - Direct Plan - Growth", "kuvera_code": "X2"},
    ]}))
    (raw / "sources.json").write_text(json.dumps({"sources": [
        {"source_id": "syn", "name": "Synthetic", "url": "https://example.invalid", "collected_at": "2026-09-30",
         "description": "synthetic", "license_notes": "test only"}
    ]}))

    start = date(2020, 1, 1)
    for code, growth in ((1, 1.0004), (2, 1.0002)):
        nav, rows = 10.0, []
        day = start
        while day <= date(2026, 9, 30):
            if day.weekday() < 5:
                rows.append({"date": day.strftime("%d-%m-%Y"), "nav": f"{nav:.4f}"})
                nav *= growth
            day += timedelta(days=1)
        rows.reverse()
        payload = {"meta": {"fund_house": "Synthetic AMC", "scheme_type": "Open Ended Schemes",
                            "scheme_category": "Equity Scheme - Flexi Cap Fund", "scheme_code": code,
                            "scheme_name": f"Synthetic {'One' if code == 1 else 'Two'} Fund - Direct Plan - Growth"},
                   "data": rows, "status": "SUCCESS"}
        (raw / "nav" / f"{code}.json").write_text(json.dumps(payload))

    # fund 1 has full attributes; fund 2 has none (-> UNKNOWN fields reported, not dropped)
    (raw / "scheme_attributes" / "1.json").write_text(json.dumps({
        "expense_ratio": "0.55", "expense_ratio_date": "2026-08-31", "aum": 125000.0,
        "crisil_rating": "Very High Risk", "sip_min": 500.0, "lump_min": 1000.0, "lock_in_period": 0,
        "investment_objective": "Synthetic objective text.", "fund_manager": "Synthetic Manager",
        "detail_info": "https://example.invalid/syn-1", "start_date": "2020-01-01", "maturity_type": "Open Ended",
    }))

    (docs / "regulatory" / "reg.md").write_text(
        "---\ndocument_id: syn-reg\ndocument_type: Regulatory\ntitle: Synthetic Regulation\nsource: Synthetic Regulator\n"
        "source_url: https://example.invalid/reg\npublication_date: 2024-01-01\nfund_id:\n---\n"
        "## Flexi Cap Fund\nA synthetic definition of a flexi cap fund for testing purposes.\n"
    )


@pytest.fixture
def pipeline_env(tmp_path, settings):
    raw, docs = tmp_path / "raw", tmp_path / "documents"
    _write_snapshot(raw, docs)
    engine = create_db_engine(database_url=f"sqlite:///{tmp_path / 'test.db'}", settings=settings)
    init_database(engine)
    return engine, raw, docs


def test_pipeline_populates_sql_computes_metrics_and_reports_quality(pipeline_env, settings):
    engine, raw, docs = pipeline_env
    Session = get_session_factory(engine)
    with Session() as session:
        report = IngestionPipeline(session, raw_dir=raw, documents_dir=docs, settings=settings).run()

    assert report.funds_loaded == 2 and report.funds_total == 2
    assert report.nav_points_loaded > 3000
    assert report.documents_loaded == 3  # 1 regulatory + 2 generated scheme summaries
    assert report.chunks_created >= 3
    # fund 2 attribute gaps are reported, never silently filled
    gaps = {(i.fund_id, i.field) for i in report.issues if i.severity == "warning"}
    assert ("SYN-2", "risk_level") in gaps and ("SYN-2", "expense_ratio") in gaps

    with Session() as session:
        repo = FundRepository(session)
        assert repo.count_funds() == 2
        f1 = repo.get_fund("SYN-1")
        f2 = repo.get_fund("SYN-2")
        assert f1.expense_ratio == 0.55 and f1.risk_level == "Very High" and f1.aum_crores == 12500.0
        assert f2.expense_ratio is None and f2.risk_level is None
        assert f1.metrics is not None and f1.metrics.cagr_3y is not None and f1.metrics.sharpe_3y is not None
        assert f1.metrics.cagr_5y is not None  # 6.7 years of history
        # structured filter keeps NULL rows (UNKNOWN) but removes definite failures
        kept = {f.fund_id for f in repo.search_funds(FundFilter(max_expense_ratio=0.6))}
        assert kept == {"SYN-1", "SYN-2"}
        assert {f.fund_id for f in repo.search_funds(FundFilter(max_expense_ratio=0.5))} == {"SYN-2"}
        chunks = repo.list_chunks(fund_ids=["SYN-1"], include_global=True)
        assert {c.metadata.fund_id for c in chunks} == {"SYN-1", None}
        assert all(c.metadata.source_url for c in chunks)

    # idempotent re-run
    with Session() as session:
        second = IngestionPipeline(session, raw_dir=raw, documents_dir=docs, settings=settings).run()
        assert second.funds_loaded == 2 and FundRepository(session).count_funds() == 2


def test_metrics_require_sufficient_history():
    short = [NavPoint(date=date(2026, 1, 1) + timedelta(days=i), nav=10 + i * 0.01) for i in range(200)]
    m = compute_fund_metrics("SYN-S", short, risk_free_rate=0.065)
    assert m.cagr_1y is None and m.cagr_3y is None and m.volatility_3y is None and m.sharpe_3y is None
    assert m.history_years is not None and m.history_years < 1

    long = [NavPoint(date=date(2019, 1, 1) + timedelta(days=i), nav=10 * (1.0003 ** i)) for i in range(0, 2900)]
    m = compute_fund_metrics("SYN-L", long, risk_free_rate=0.065)
    assert m.cagr_1y is not None and m.cagr_3y is not None and m.cagr_5y is not None
    assert m.volatility_3y is not None and m.volatility_3y >= 0
    assert m.max_drawdown_3y is not None and m.max_drawdown_3y <= 0
    assert m.rolling_3y_positive_pct == 1.0  # monotonically increasing synthetic series
    assert m.risk_free_rate_used == 0.065
