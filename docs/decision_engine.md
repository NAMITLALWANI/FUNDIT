# Decision Engine

The decision engine is the deterministic core of the system. It runs **before** any LLM call,
operates **only** on the candidate set produced by SQL filtering and bounded evidence retrieval,
and produces the winner, the ranking, the trade-offs, the confidence signals and the abstention
decision. Gemini later explains this result; it cannot change it.

Code: `app/query/` (understanding), `app/decision/` (constraints, scoring, trade-offs,
confidence, sensitivity), `app/evidence/validator.py` (evidence binding).

---

## 1. Query understanding

`QueryAnalyzer` → `ConstraintExtractor` (regex / rule based) → `DecisionQuery` (Pydantic).

| Extracted | Example | Stored as |
|---|---|---|
| Amount + frequency | "₹5,000 per month", "2 lakh lump sum" | `investment_amount`, `investment_frequency`, `constraints.sip_amount` or `constraints.lump_sum_amount` |
| Horizon | "for 5 years", "3 to 5 years" (lower bound + warning) | `horizon_years`, `constraints.max_lock_in_years` |
| Risk tolerance | "moderate risk", "don't want very high volatility" | `constraints.max_risk_level` (SEBI label ceiling) |
| Category | "large cap", "ELSS", "index fund", "debt" | `constraints.categories` / `sub_categories` |
| Cost limit | "expense ratio under 0.5%" | `constraints.max_expense_ratio` |
| Exit load / AUM / return floor | "no exit load", "AUM above 5000 crore", "at least 12% return" | `exit_load_free`, `min_aum_crores`, `min_return_3y` |
| Plan type | "direct plan" | `constraints.plan_type` — **never assumed** |
| AMC | "prefer HDFC funds" (soft) vs "only HDFC funds" (hard) | `preferences.preferred_amcs` vs `constraints.required_amcs` |
| Preferences | "low cost", "stable", "consistent", "capital appreciation" | `preferences.weights` (0..1 emphasis) |
| Named funds | "Compare Parag Parikh Flexi Cap and HDFC Flexi Cap" | `comparison_targets` resolved against the `funds` table |

### Ambiguity handling (no silent guessing)

Anything unresolved is an explicit `Ambiguity(field, message, severity)`:

* `warning` – an interpretation was made and is disclosed (e.g. colloquial risk mapping, horizon range).
* `blocking` – the request cannot be evaluated; the engine abstains (guaranteed/risk-free returns,
  a named fund that does not exist in the database, a query with no extractable criteria).

An amount without "monthly"/"lump sum" wording is recorded as a warning and **no**
minimum-investment constraint is applied rather than guessing.

### Risk vocabulary → SEBI Risk-o-meter ceiling

SEBI uses six levels; users use three words. Each colloquial tier covers two SEBI levels and the
*ceiling* of the tier is used, so a "moderate" investor is never shown "High" funds:

| User says | Ceiling | Exact SEBI label? |
|---|---|---|
| low / conservative | Low to Moderate | no → warning |
| moderate / medium / balanced | Moderately High | no → warning |
| high / aggressive | Very High | no → warning |
| zero / no / minimal risk | Low | no → warning |
| "very high", "moderately high", "low to moderate" | as stated | yes |
| "don't want very high volatility" | High | no → warning |

Consequence worth knowing: almost all diversified equity funds are labelled **Very High** on the
Risk-o-meter, so "moderate risk" legitimately restricts the candidate set to debt and
conservative hybrid schemes. The system reports this interpretation instead of overriding the
regulator's label.

---

## 2. Hard constraints: PASS / FAIL / UNKNOWN

`app/decision/constraints.py` evaluates only the constraints the user set, against structured
data (`funds`, `fund_metrics`). For each constraint:

* value present and satisfies → `PASS`
* value present and violates → `FAIL`
* value `NULL` in the database → `UNKNOWN` — **never PASS**

Aggregate: any `FAIL` → `FAIL`; else any `UNKNOWN` → `UNKNOWN`; else `PASS`.

SQL pre-filtering (`FundRepository.search_funds`) removes only *definite* failures and keeps
`NULL` rows so the engine can label them `UNKNOWN`. `FAIL` candidates are excluded from the
ranking (listed with reasons); `UNKNOWN` candidates are ranked **below every PASS candidate**
and can never be the winner. If no PASS candidate exists the system abstains.

---

## 3. Utility scoring

All component scores are min-max normalised **within the candidate set** (never the whole
catalogue). Weights live in `Settings` (`.env`) and must sum to 1.0.

| Component | Default weight | Basis | Rationale |
|---|---|---|---|
| `risk_adjusted_return` | 0.35 | `fund_metrics.sharpe_3y` | The user's objective is return per unit of risk; Sharpe is the standard, auditable summary and the single most informative structured metric available. |
| `expense_efficiency` | 0.20 | `funds.expense_ratio` | Cost is the one input that is known with certainty and compounds directly against returns; SEBI publishes it daily. |
| `downside_protection` | 0.15 | `volatility_3y`, `max_drawdown_3y` | Captures the risk experience separately from Sharpe so a fund with high Sharpe but deep drawdowns is not over-rewarded. |
| `preference_fit` | 0.15 | user soft preferences | Lets stated priorities (low cost, stability, consistency, preferred AMC…) reorder candidates without eliminating any. Absent when no preference is stated (weight renormalised). |
| `evidence_quality` | 0.10 | verified document chunks (coverage × relevance) | Recommendations must be explainable from documents; missing evidence contributes **0 while keeping the weight** (a penalty, never a default score). |
| `aum_context` | 0.05 | `funds.aum_crores` | Size is contextual (liquidity, closure risk) but not a quality signal; deliberately the smallest weight. |

Additional rules:

* **Missing structured data** → component `None`, weight dropped from the denominator; the gap is
  exposed via `data_completeness` and lowers confidence rather than silently moving the rank.
* **UNKNOWN penalty** → `unknown_constraint_penalty` (default 0.10) per UNKNOWN constraint.
* **Return window** → `cagr_5y` is used for the `high_return` preference when the horizon is
  ≥ 5 years, otherwise `cagr_3y`.

Every `CandidateScore` carries its `ComponentScore` list (weight, score, contribution, basis) and
the `StructuredFact` list used, so any number in the output can be traced to a table and an
as-of date.

### Why these numbers and not others

There is no ground-truth "correct" weight vector for a decision-support ranking; the defaults
encode the ordering *return-per-risk > cost > downside > stated preferences > evidence > size*
and are validated two ways: (1) they are configuration, so a reviewer can change them without
code; (2) the **sensitivity analysis** below reports how fragile any specific decision is to them.

---

## 4. Sensitivity (ranking stability)

`SensitivityAnalyzer` re-runs scoring with each weight scaled by ±20 % (renormalised) — 12
deterministic trials — and reports:

* `winner_flip_rate` – share of trials where the top PASS candidate changed,
* `top3_jaccard_mean` – stability of the shortlist,
* `alternative_winners` – who won in the flipped trials.

`ranking_stability = 1 − winner_flip_rate` feeds the confidence composite and the abstention gate.

---

## 5. Trade-offs

`TradeoffAnalyzer` compares the winner with up to three alternatives using **structured facts
only** (expense ratio, Sharpe, CAGR, volatility, drawdown, rolling consistency, AUM, minimum SIP,
Risk-o-meter level). A difference is reported only if it exceeds both a relative and an absolute
threshold; fields missing on either side are listed as "not comparable". Each `Tradeoff` records
its `fact_basis`.

---

## 6. Confidence and abstention

Confidence is a **system/decision confidence signal**, not a probability of future returns.
Seven signals in [0, 1], equal-weighted (documented simplification: no empirical basis exists in
this project to rank signals):

| Signal | Meaning |
|---|---|
| `constraint_completeness` | share of the winner's user-set constraints that are PASS |
| `evidence_coverage` | verified chunks for the winner ÷ `evidence_chunks_per_fund` |
| `evidence_quality` | winner's evidence component |
| `score_margin` | `min(1, margin / (3 × min_score_margin))` |
| `data_completeness` | share of scoring components with data for the winner |
| `data_freshness` | 1 inside `data_freshness_days`, linear decay to 0 at 2× |
| `ranking_stability` | 1 − winner flip rate |

Level: composite ≥ `confidence_high_threshold` (0.70) → **HIGH**; ≥ `confidence_medium_threshold`
(0.45) → **MEDIUM**; else **LOW**. Underlying signals and human-readable reasons are always returned.

### Abstention gate (evaluated in order)

1. Any `blocking` ambiguity in the query.
2. No PASS candidate (all FAIL, or only UNKNOWN remain).
3. Winner has fewer than `min_evidence_chunks_for_generation` verified chunks.
4. Winner margin < `min_score_margin` **and** the winner flipped in at least one sensitivity trial
   (close *and* unstable). A close but perfectly stable margin only lowers confidence.
5. Optional policy: `abstain_on_low_confidence=true` abstains on LOW.

When the gate fires, `DecisionResult.winner` is `None`, `abstention_reason` explains why, and the
LLM is never called.

---

## 7. Evidence validation

`EvidenceValidator` binds reranked chunks to candidates before scoring:

* chunks for funds **outside the candidate set** are discarded (they can never become citations);
* `fund_id = NULL` chunks (SEBI circulars) become shared context, not fund evidence;
* numeric statements in generated scheme summaries (expense ratio, AUM, minimum SIP, risk level)
  are cross-checked against the database row; mismatches mark the chunk `CONFLICTING` and it is
  withheld from the fund's evidence;
* publication dates older than `data_freshness_days` are counted as stale.

Document text never overrides a structured value; conflicts are surfaced, not resolved.

---

## 8. Known limitations (Step 3)

* Evidence retrieval returns a single top-K across all candidates, so with many candidates some
  funds receive no chunks and the engine must abstain or rank them down. The workflow (Step 4)
  adds a per-fund evidence top-up for the leading candidates before the final decision.
* `exit_load` and `benchmark` are not present in the public snapshot, so constraints on them
  always evaluate to `UNKNOWN` (by design, not as a workaround).
* Horizon is used only for lock-in checks and the return window; the engine does not impose
  asset-class suitability rules (that would be advice, not evidence).
