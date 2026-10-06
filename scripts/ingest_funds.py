"""
Populate the relational database from the public data snapshot in data/raw/.

Usage:
    python scripts/ingest_funds.py                 # uses DATABASE_URL from settings/.env
    python scripts/ingest_funds.py --reset         # drop and recreate all tables first
    python scripts/ingest_funds.py --report-path data/processed/data_quality_report.json

Safe to re-run: funds, metrics and documents are upserted; NAV history is replaced per fund.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.db.session import create_db_engine, get_session_factory, init_database  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--documents-dir", default=str(ROOT / "data" / "documents"))
    parser.add_argument("--report-path", default=str(ROOT / "data" / "processed" / "data_quality_report.json"))
    parser.add_argument("--reset", action="store_true", help="Drop all tables before ingesting")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, json_format=settings.log_json)

    engine = create_db_engine(settings=settings)
    if args.reset:
        Base.metadata.drop_all(engine)
    init_database(engine)

    session_factory = get_session_factory(engine)
    with session_factory() as session:
        pipeline = IngestionPipeline(
            session=session,
            raw_dir=Path(args.raw_dir),
            documents_dir=Path(args.documents_dir),
            settings=settings,
        )
        report = pipeline.run()

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    by_severity = {"error": 0, "warning": 0, "info": 0}
    for issue in report.issues:
        by_severity[issue.severity] += 1

    print("\n=== Data quality report ===")
    print(f"Funds loaded:               {report.funds_loaded}/{report.funds_total}")
    print(f"Funds with attribute gaps:  {report.funds_with_attribute_gaps}")
    print(f"NAV points loaded:          {report.nav_points_loaded}")
    print(f"Documents / chunks:         {report.documents_loaded} / {report.chunks_created}")
    print(f"Issues (error/warn/info):   {by_severity['error']} / {by_severity['warning']} / {by_severity['info']}")
    print(f"Full report written to:     {report_path}")
    if by_severity["error"]:
        print("\nErrors:")
        for issue in report.issues:
            if issue.severity == "error":
                print(f"  - [{issue.fund_id}] {issue.field}: {issue.message}")
    return 1 if by_severity["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
