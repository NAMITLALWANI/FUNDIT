#!/usr/bin/env python3
"""
CLI script to run Information Retrieval & Grounded Generation benchmark evaluation.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import get_logger, setup_logging
from app.evaluation.evaluator import BenchmarkEvaluator

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Decision-Augmented RAG Engine")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="data/evaluation/queries.json",
        help="Path to evaluation queries JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save full evaluation JSON report",
    )
    args = parser.parse_args()

    setup_logging(log_level="INFO")
    print("=" * 80)
    print("AI DECISION ENGINE V2 - INFORMATION RETRIEVAL & GROUNDING EVALUATION REPORT")
    print("=" * 80)

    evaluator = BenchmarkEvaluator()
    report = evaluator.evaluate_benchmark(
        benchmark_file=args.benchmarks,
    )

    rm = report.retrieval_metrics
    gm = report.generation_metrics

    print("\n[RETRIEVAL METRICS]")
    print(f"  Recall@1:    {rm.recall_at_k.get(1, 0.0):.4f}")
    print(f"  Recall@3:    {rm.recall_at_k.get(3, 0.0):.4f}")
    print(f"  Recall@5:    {rm.recall_at_k.get(5, 0.0):.4f}")
    print(f"  Precision@1: {rm.precision_at_k.get(1, 0.0):.4f}")
    print(f"  Precision@3: {rm.precision_at_k.get(3, 0.0):.4f}")
    print(f"  Precision@5: {rm.precision_at_k.get(5, 0.0):.4f}")
    print(f"  MRR:         {rm.mrr:.4f}")
    print(f"  NDCG@3:      {rm.ndcg_at_k.get(3, 0.0):.4f}")
    print(f"  NDCG@5:      {rm.ndcg_at_k.get(5, 0.0):.4f}")

    print("\n[GENERATION & DECISION METRICS]")
    print(f"  Citation Coverage:       {gm.citation_coverage:.4f}")
    print(f"  Constraint Faithfulness: {gm.constraint_faithfulness:.4f}")
    print(f"  Answer Relevance:        {gm.answer_relevance:.4f}")

    print("\n[PER-QUERY BREAKDOWN]")
    for q in report.query_details:
        print(f"  • [{q['query_id']}]")
        print(f"    Query:    {q['query']}")
        print(f"    Winner:   {q['winner']}")
        print(f"    Expected: {q['expected_funds']}")
        print(f"    Found:    {q['retrieved_funds']}")
        print(f"    Recall@3: {q['recall@3']:.2f} | MRR: {q['mrr']:.2f} | NDCG@3: {q['ndcg@3']:.2f} | Faithfulness: {q['constraint_faithfulness']:.2f}")

    print("=" * 80)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)
        print(f"Full report exported to: {args.output}")


if __name__ == "__main__":
    main()
