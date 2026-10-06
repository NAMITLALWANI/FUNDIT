"""
System and user prompts enforcing strict grounded generation and citation constraints.

Financial safety rules embedded in the system prompt:
  - The LLM is not the decision maker; it explains an already-computed result.
  - Every factual claim must cite an evidence marker ([1], [2], ...).
  - Return probability and investment guarantees are forbidden.
  - Output must be strict JSON matching GeneratedDecision.
"""

from typing import List, Optional

from app.decision.models import DecisionResult
from app.generation.models import Citation
from app.query.models import DecisionQuery


DECISION_SYSTEM_PROMPT = """You are the AI Decision Engine Assistant for an Evidence-Driven Mutual Fund Decision Support System.

Your role is strictly to EXPLAIN a decision that has already been made by a deterministic scoring engine.
You are NOT the decision maker. You CANNOT change the winner, rankings, or scores.

STRICT OPERATIONAL RULES:
1. ONLY make factual claims directly substantiated by the provided verified evidence chunks.
2. ALWAYS attach numbered citation markers (e.g., [1], [2]) after each factual statement you make.
3. NEVER fabricate, extrapolate, or hallucinate fund performance data, expense ratios, AUM figures, returns, or any financial metrics.
4. NEVER promise, imply, or suggest guaranteed future returns. Mutual funds do not guarantee returns.
5. RESPECT the deterministic winner and rankings provided. Do not change them.
6. If evidence is insufficient to support a claim, say so explicitly rather than guessing.
7. OUTPUT STRICT JSON ONLY with keys: "recommendation_text", "summary", "reasons", "tradeoff_notes", "cited_chunk_ids".
8. "cited_chunk_ids" must list only the chunk IDs from the provided evidence that you actually referenced.
9. Do not include any preamble, chain-of-thought, or text outside the JSON object.

FINANCIAL DISCLAIMER (always reflect this perspective):
This analysis is based on historical data and publicly available scheme documents.
Past performance does not guarantee future results. This is decision-support, not financial advice.
"""


def format_evidence_block(citations: List[Citation]) -> str:
    """Format citation list as numbered evidence context for the LLM prompt."""
    if not citations:
        return "No verified evidence chunks were retrieved for this query."
    blocks = []
    for c in citations:
        block = (
            f"Evidence [{c.citation_id}]\n"
            f"Chunk ID: {c.chunk_id}\n"
            f"Source: {c.source} — {c.title}\n"
            f"Document Type: {c.document_type}\n"
            f"Fund: {c.fund_id or 'General / Regulatory'}\n"
            f"URL: {c.source_url or 'N/A'}\n"
            f"Excerpt: {c.snippet}\n"
        )
        blocks.append(block)
    return "\n---\n".join(blocks)


def format_user_prompt(
    query: DecisionQuery,
    decision: DecisionResult,
    citations: List[Citation],
) -> str:
    """Construct structured user prompt containing the decision context and evidence."""
    winner_str = "ABSTAINED — No eligible fund could be recommended."
    if decision.winner:
        w = decision.winner
        facts = {f.field: f.value for f in w.structured_facts}
        winner_str = (
            f"Fund: {w.fund_name} (AMC: {w.amc})\n"
            f"Category: {w.category}" + (f" / {w.sub_category}" if w.sub_category else "") + "\n"
            f"Constraint Status: {w.constraint_status}\n"
            f"Final Score: {w.final_score:.4f} (rank 1 of {len(decision.ranked)})\n"
            f"Expense Ratio: {facts.get('expense_ratio', 'UNKNOWN')}%\n"
            f"AUM: Rs {facts.get('aum_crores', 'UNKNOWN'):,.0f} crore\n"
            f"3-year Sharpe: {facts.get('sharpe_3y', 'UNKNOWN')}\n"
            f"3-year CAGR: {'{:.1%}'.format(facts['cagr_3y']) if facts.get('cagr_3y') is not None else 'UNKNOWN'}\n"
            f"Risk Level: {facts.get('risk_level', 'UNKNOWN')}\n"
        )

    tradeoff_str = ""
    if decision.tradeoffs:
        parts = []
        for t in decision.tradeoffs[:3]:
            parts.append(
                f"vs {t.fund_b}:\n"
                f"  Winner advantages: {'; '.join(t.advantages_a) or 'None identified'}\n"
                f"  Alternative advantages: {'; '.join(t.advantages_b) or 'None identified'}"
            )
        tradeoff_str = "\n\n".join(parts)
    else:
        tradeoff_str = "No trade-off data available."

    selection_str = "\n".join(f"- {r}" for r in decision.selection_reasons) or "N/A"
    ambiguity_str = (
        "\n".join(f"- [{a.severity.upper()}] {a.message}" for a in query.ambiguities)
        if query.ambiguities
        else "None"
    )

    evidence_str = format_evidence_block(citations)

    prompt = f"""### USER QUERY
{query.raw_query}

### PARSED INVESTMENT CONSTRAINTS
- Investment Amount: {query.investment_amount or 'Not specified'} {query.investment_frequency or ''}
- Horizon: {'{} years'.format(query.horizon_years) if query.horizon_years else 'Not specified'}
- Risk Tolerance: {query.risk_tolerance or 'Not specified'}
- Objective: {query.objective or 'Not specified'}
- Hard Constraints: {query.constraints.model_dump(exclude_none=True)}
- Soft Preferences: {list(query.preferences.weights.keys()) or 'None'}
- Query Ambiguities: {ambiguity_str}

### DETERMINISTIC DECISION RESULT
Confidence Level: {decision.confidence.level} ({decision.confidence.composite:.2f}/1.0)
Candidates Evaluated: {decision.candidates_considered}

WINNER:
{winner_str}

KEY SELECTION REASONS (from the scoring engine):
{selection_str}

TRADE-OFFS:
{tradeoff_str}

### VERIFIED EVIDENCE CHUNKS
{evidence_str}

### INSTRUCTIONS
Generate a JSON response explaining this decision. Use citation markers like [1], [2] after factual statements.
Your JSON must have exactly these keys: "recommendation_text", "summary", "reasons", "tradeoff_notes", "cited_chunk_ids".
Do not change the winner or scores. Do not promise future returns.
"""
    return prompt
