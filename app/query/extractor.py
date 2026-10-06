"""
Deterministic constraint, preference and target extraction for mutual fund queries.

Design rules
* Hard constraints are only set when the user states them. Nothing is defaulted (e.g. plan type,
  SIP vs lump sum) - unresolved items become explicit ``Ambiguity`` records.
* Colloquial risk vocabulary (low / moderate / high) is mapped to a SEBI Risk-o-meter *ceiling*
  using the documented 3-tier -> 6-level table below; the interpretation is surfaced as a warning.
* Named funds are resolved against the fund table by token overlap; unresolved names that look
  like fund references (contain an AMC token) are blocking ambiguities rather than guesses.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from app.ingestion.models import RISK_LEVEL_ORDER
from app.query.models import (
    Ambiguity,
    ComparisonTarget,
    QueryConstraints,
    UserPreferences,
)

# Colloquial tier -> highest acceptable SEBI label. Each colloquial tier covers two SEBI levels;
# the ceiling of the tier is used so a 'moderate' investor is not shown 'High' funds.
RISK_VOCABULARY: Dict[str, Tuple[str, bool]] = {
    # phrase: (ceiling label, is_exact_sebi_label)
    "very high": ("Very High", True),
    "moderately high": ("Moderately High", True),
    "low to moderate": ("Low to Moderate", True),
    "aggressive": ("Very High", False),
    "high": ("Very High", False),
    "moderate": ("Moderately High", False),
    "medium": ("Moderately High", False),
    "balanced": ("Moderately High", False),
    "conservative": ("Low to Moderate", False),
    "low": ("Low to Moderate", False),
    "minimal": ("Low", False),
    "zero": ("Low", False),
    "no": ("Low", False),
}

SUB_CATEGORY_KEYWORDS: Dict[str, str] = {
    r"\blarge[\s-]?cap\b|\bbluechip\b|\bblue[\s-]chip\b": "Large Cap Fund",
    r"\bmid[\s-]?cap\b": "Mid Cap Fund",
    r"\bsmall[\s-]?cap\b": "Small Cap Fund",
    r"\bflexi[\s-]?cap\b|\bmulti[\s-]?cap\b": "Flexi Cap Fund",
    r"\belss\b|\btax[\s-]?sav(?:er|ing)\b|\b80c\b": "ELSS",
    r"\bindex fund\b|\bnifty\b|\bpassive\b|\bsensex\b": "Index Fund",
    r"\bliquid fund\b|\bliquid\b": "Liquid Fund",
    r"\bgilt\b|\bgovernment securit": "Gilt Fund",
    r"\bcorporate bond\b": "Corporate Bond Fund",
    r"\bshort[\s-]?(?:term|duration)\s+(?:debt|fund|bond)": "Short Duration Fund",
    r"\bbalanced advantage\b|\bdynamic asset allocation\b": "Dynamic Asset Allocation or Balanced Advantage",
    r"\bconservative hybrid\b": "Conservative Hybrid Fund",
    r"\bcontra\b|\bcontrarian\b": "Contra Fund",
    r"\bsectoral\b|\bthematic\b|\btechnology fund\b|\btech fund\b|\bdigital\b|\bit sector\b": "Sectoral/Thematic",
}

CATEGORY_KEYWORDS: Dict[str, str] = {
    r"\bequity\b|\bstock market\b": "Equity",
    r"\bdebt\b|\bbond fund\b|\bfixed income\b": "Debt",
    r"\bhybrid\b": "Hybrid",
}

PREFERENCE_PATTERNS: Dict[str, str] = {
    "low_cost": r"\blow(?:er|est)?[\s-]?(?:cost|expense|fee|ter)\b|\bcheap(?:er|est)?\b|\bexpense ratio\b|\bcost[\s-]?efficient\b",
    "low_volatility": r"\blow(?:er)?[\s-]?volatil|\bless volatil|\bstable\b|\bstability\b|\bavoid volatil|\bwithout (?:very )?high volatil|\bdon'?t want (?:very )?high volatil|\bnot (?:too |very )?volatile\b|\bdownside\b|\bcapital (?:protection|preservation)\b",
    "high_return": r"\bhigh(?:er|est)?[\s-]?(?:return|growth)|\bbest[\s-]?return|\bmaximi[sz]e returns?\b|\bcapital appreciation\b|\bwealth creation\b|\bgrowth\b|\baggressive growth\b",
    "consistency": r"\bconsisten(?:t|cy)\b|\bsteady\b|\breliabl(?:e|y)\b|\btrack record\b",
    "fund_size": r"\blarge (?:fund|aum)\b|\bestablished\b|\bbig(?:ger)? fund\b|\bfund size\b",
}

OBJECTIVE_PATTERNS: Dict[str, str] = {
    "retirement": r"\bretire",
    "tax_saving": r"\btax[\s-]?sav|\b80c\b|\belss\b",
    "emergency_fund": r"\bemergency\b|\bparking\b|\bpark (?:my )?(?:money|cash)",
    "regular_income": r"\bregular income\b|\bmonthly income\b",
    "child_education": r"\bchild(?:ren)?'?s? (?:education|future)\b|\beducation\b",
    "capital_preservation": r"\bcapital (?:preservation|protection)\b",
    "capital_appreciation": r"\bcapital appreciation\b|\bwealth creation\b|\blong[\s-]term growth\b",
}

AMOUNT_RE = re.compile(
    r"(?:₹|rs\.?|inr|rupees?)?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(k|thousand|lakhs?|lacs?|l\b|crores?|cr\b)?\s*(?:₹|rs\.?|inr|rupees)?",
    re.IGNORECASE,
)
MONTHLY_RE = re.compile(r"\bper month\b|/\s*month|\bmonthly\b|\bevery month\b|\ba month\b|\bsip\b|\bp\.?m\.?\b", re.IGNORECASE)
LUMP_SUM_RE = re.compile(r"\blump[\s-]?sum\b|\bone[\s-]?time\b|\bat once\b|\bone shot\b|\bsingle investment\b", re.IGNORECASE)
HORIZON_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|to|–)?\s*(\d+(?:\.\d+)?)?\s*(?:-|\s)?\s*(?:years?|yrs?|y\b)",
    re.IGNORECASE,
)
MONTH_HORIZON_RE = re.compile(r"(\d+)\s*(?:-|\s)?\s*months?\b", re.IGNORECASE)
EXPENSE_RE = re.compile(
    r"(?:expense ratio|ter|expense|cost)[^.%\d]{0,25}?(?:under|below|less than|max(?:imum)?|up to|upto|<=?|of|at most)?\s*(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
EXPENSE_RE_ALT = re.compile(
    r"(?:under|below|less than|max(?:imum)?|up to|upto|at most)?\s*(\d+(?:\.\d+)?)\s*%\s*(?:expense ratio|ter\b|expense|cost)",
    re.IGNORECASE,
)
MIN_RETURN_RE = re.compile(
    r"(?:at least|minimum|min|above|more than|over|guaranteed|assured)\s*(\d+(?:\.\d+)?)\s*%\s*(?:annual(?:i[sz]ed)?\s*)?(?:returns?|cagr|growth)",
    re.IGNORECASE,
)
GUARANTEE_RE = re.compile(r"\bguarantee[ds]?\b|\bassured returns?\b|\brisk[\s-]free returns?\b|\bno risk\b|\bzero risk\b", re.IGNORECASE)
MIN_AUM_RE = re.compile(r"aum\s*(?:above|over|at least|more than|>=?)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(crores?|cr\b)?", re.IGNORECASE)
EXIT_LOAD_FREE_RE = re.compile(r"\b(?:no|zero|without|nil)\s+exit[\s-]?load\b|\bexit[\s-]?load[\s-]?free\b|\bexit load waiver\b", re.IGNORECASE)
PLAN_RE = re.compile(r"\b(direct|regular)\s+plan\b", re.IGNORECASE)
ONLY_AMC_RE = re.compile(r"\bonly\s+(?:from\s+)?([a-z][a-z\s&]+?)\s+(?:funds?|schemes?)\b", re.IGNORECASE)
PREFER_AMC_RE = re.compile(r"\bprefer(?:ably|red)?\s+(?:funds?\s+from\s+)?([a-z][a-z\s&]+?)\s+(?:funds?|schemes?|amc)\b", re.IGNORECASE)
COMPARE_RE = re.compile(
    r"(?:compare|comparison between|difference between|rank(?:ed)?)\s+(.+?)\s+(?:and|vs\.?|versus|with|against|above|over|below)\s+(.+?)(?=\s+for\b|\s+over\b|\s+on\b|\s+in\b|[.?!,;]|$)",
    re.IGNORECASE,
)

NAME_NOISE_TOKENS = {
    "fund", "funds", "direct", "regular", "plan", "growth", "option", "the", "mutual", "scheme",
    "erstwhile", "idcw", "dividend", "payout", "reinvestment", "india", "asset", "prudential",
    "mahindra", "oswal", "robeco", "templeton", "-", "&",
}
AMC_TOKENS = {
    "mirae", "icici", "canara", "nippon", "sbi", "parag", "parikh", "ppfas", "hdfc", "franklin",
    "kotak", "quant", "axis", "dsp", "uti", "motilal", "edelweiss", "tata", "invesco", "aditya",
    "birla", "bandhan", "sundaram", "lic", "baroda", "bnp", "hsbc", "pgim", "whiteoak", "navi",
    "zerodha", "groww", "360", "jm", "mahindra", "manulife", "union", "shriram", "helios", "nj",
}
UNIT_MULTIPLIERS = {
    "k": 1_000.0, "thousand": 1_000.0,
    "lakh": 100_000.0, "lakhs": 100_000.0, "lac": 100_000.0, "lacs": 100_000.0, "l": 100_000.0,
    "crore": 10_000_000.0, "crores": 10_000_000.0, "cr": 10_000_000.0,
}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _fund_key_tokens(fund_name: str) -> List[str]:
    base = re.sub(r"\(.*?\)", " ", fund_name)
    return [t for t in _tokens(base) if t not in NAME_NOISE_TOKENS]


@dataclass
class ExtractionResult:
    investment_amount: Optional[float]
    investment_frequency: Optional[str]
    horizon_years: Optional[float]
    risk_tolerance: Optional[str]
    objective: Optional[str]
    constraints: QueryConstraints
    preferences: UserPreferences
    comparison_targets: List[ComparisonTarget]
    ambiguities: List[Ambiguity]
    intent: str


class FundNameResolver:
    """Resolve free-text fund mentions to fund IDs using key-token overlap."""

    def __init__(self, known_funds: Sequence[Tuple[str, str]], threshold: float = 0.75) -> None:
        self.threshold = threshold
        self.entries = [
            (fund_id, name, _fund_key_tokens(name)) for fund_id, name in known_funds
        ]

    def resolve(self, mention: str) -> Tuple[Optional[ComparisonTarget], List[str]]:
        """Return (target or None, candidate names when ambiguous)."""
        mention_tokens = set(_tokens(mention))
        if not mention_tokens:
            return None, []
        scored: List[Tuple[float, str, str]] = []
        for fund_id, name, key_tokens in self.entries:
            if not key_tokens or key_tokens[0] not in mention_tokens:
                continue
            overlap = sum(1 for t in key_tokens if t in mention_tokens)
            score = overlap / len(key_tokens)
            if overlap >= 2 and score >= self.threshold:
                scored.append((score, fund_id, name))
        if not scored:
            return None, []
        scored.sort(key=lambda s: (-s[0], s[2]))
        best = scored[0]
        ties = [s for s in scored if abs(s[0] - best[0]) < 1e-9]
        if len(ties) > 1:
            return None, [s[2] for s in ties]
        return ComparisonTarget(mention=mention.strip(), fund_id=best[1], fund_name=best[2], match_score=round(best[0], 3)), []

    def find_all(self, query: str) -> List[ComparisonTarget]:
        """Find every fund explicitly named anywhere in the query (highest score first, no duplicates)."""
        query_tokens = set(_tokens(query))
        hits: List[ComparisonTarget] = []
        for fund_id, name, key_tokens in self.entries:
            if not key_tokens or key_tokens[0] not in query_tokens:
                continue
            overlap = sum(1 for t in key_tokens if t in query_tokens)
            score = overlap / len(key_tokens)
            if overlap >= 2 and score >= self.threshold:
                hits.append(ComparisonTarget(mention=name, fund_id=fund_id, fund_name=name, match_score=round(score, 3)))
        hits.sort(key=lambda h: (-h.match_score, h.fund_name or ""))
        return hits


class ConstraintExtractor:
    """Regex/rule-based extractor producing validated constraints, preferences and ambiguities."""

    def __init__(self, known_funds: Optional[Sequence[Tuple[str, str]]] = None) -> None:
        self.resolver = FundNameResolver(known_funds or [])

    # ------------------------------------------------------------------ public

    def extract(self, query: str) -> ExtractionResult:
        q = query.strip()
        ql = q.lower()
        ambiguities: List[Ambiguity] = []

        amount, frequency = self._extract_amount(ql, ambiguities)
        horizon = self._extract_horizon(ql, ambiguities)
        risk = self._extract_risk(ql, ambiguities)
        categories, sub_categories = self._extract_categories(ql)
        max_expense = self._extract_expense(ql)
        min_return = self._extract_min_return(ql, ambiguities)
        min_aum = self._extract_min_aum(ql)
        preferences = self._extract_preferences(ql, max_expense)
        objective = self._extract_objective(ql)
        plan_type = self._extract_plan(ql)
        required_amcs, preferred_amcs = self._extract_amcs(ql)
        if preferred_amcs:
            preferences.preferred_amcs = preferred_amcs
            preferences.weights.setdefault("preferred_amc", 0.6)

        if GUARANTEE_RE.search(ql) and not re.search(r"\b(?:no|not|never)\s+guarantee", ql):
            ambiguities.append(
                Ambiguity(
                    field="guaranteed_return",
                    message="The query asks for guaranteed or risk-free returns. Mutual funds cannot guarantee returns; this requirement cannot be evaluated.",
                    severity="blocking",
                )
            )

        constraints = QueryConstraints(
            sip_amount=amount if frequency == "monthly_sip" else None,
            lump_sum_amount=amount if frequency == "lump_sum" else None,
            max_expense_ratio=max_expense,
            max_risk_level=risk,
            categories=categories or None,
            sub_categories=sub_categories or None,
            min_aum_crores=min_aum,
            max_lock_in_years=horizon if horizon is not None else None,
            exit_load_free=True if EXIT_LOAD_FREE_RE.search(ql) else None,
            required_amcs=required_amcs or None,
            plan_type=plan_type,
            min_return_3y=min_return,
        )

        targets, intent = self._extract_targets(q, ql, ambiguities)
        if targets:
            # Explicitly named funds define the candidate set; category words inside the fund
            # names must not become additional filters.
            constraints = constraints.model_copy(update={"categories": None, "sub_categories": None})

        return ExtractionResult(
            investment_amount=amount,
            investment_frequency=frequency,
            horizon_years=horizon,
            risk_tolerance=risk,
            objective=objective,
            constraints=constraints,
            preferences=preferences,
            comparison_targets=targets,
            ambiguities=ambiguities,
            intent=intent,
        )

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _to_number(raw: str, unit: Optional[str]) -> float:
        value = float(raw.replace(",", ""))
        if unit:
            value *= UNIT_MULTIPLIERS.get(unit.lower().rstrip("."), 1.0)
        return value

    def _extract_amount(self, ql: str, ambiguities: List[Ambiguity]) -> Tuple[Optional[float], Optional[str]]:
        # Remove horizon/percent fragments so '5 years' or '1%' are not read as amounts.
        scrubbed = HORIZON_RE.sub(" ", ql)
        scrubbed = MONTH_HORIZON_RE.sub(" ", scrubbed)
        scrubbed = re.sub(r"\d+(?:\.\d+)?\s*%", " ", scrubbed)
        scrubbed = re.sub(r"\bnifty\s*\d+\b|\bsensex\s*\d*\b|\b80c\b|\btop\s*\d+\b", " ", scrubbed)

        amount: Optional[float] = None
        for match in AMOUNT_RE.finditer(scrubbed):
            raw, unit = match.group(1), match.group(2)
            has_currency = bool(re.search(r"₹|rs\.?|inr|rupees?", match.group(0), re.IGNORECASE))
            if not unit and not has_currency and len(raw.replace(",", "")) < 4:
                continue  # bare small integers ('2 funds', 'top 3') are not amounts
            value = self._to_number(raw, unit)
            if value < 100:
                continue
            amount = value
            break

        frequency: Optional[str] = None
        if MONTHLY_RE.search(ql):
            frequency = "monthly_sip"
        if LUMP_SUM_RE.search(ql):
            frequency = "lump_sum" if frequency is None else frequency
            if MONTHLY_RE.search(ql):
                ambiguities.append(Ambiguity(field="investment_frequency", message="Both monthly and lump-sum wording found; the monthly SIP interpretation was used.", severity="warning"))

        if amount is not None and frequency is None:
            ambiguities.append(
                Ambiguity(
                    field="investment_frequency",
                    message=f"An amount of Rs {amount:,.0f} was found but it is unclear whether it is a monthly SIP or a lump sum; the minimum-investment constraint was not applied.",
                    severity="warning",
                    candidates=["monthly_sip", "lump_sum"],
                )
            )
        return amount, frequency

    @staticmethod
    def _extract_horizon(ql: str, ambiguities: List[Ambiguity]) -> Optional[float]:
        for match in HORIZON_RE.finditer(ql):
            lo = float(match.group(1))
            hi = float(match.group(2)) if match.group(2) else None
            if lo > 60:
                continue
            if hi is not None and hi != lo:
                ambiguities.append(Ambiguity(field="horizon_years", message=f"Horizon given as a range ({lo:g}-{hi:g} years); the lower bound {lo:g} years was used for lock-in checks.", severity="warning"))
            return lo
        month_match = MONTH_HORIZON_RE.search(ql)
        if month_match:
            return round(int(month_match.group(1)) / 12.0, 2)
        if re.search(r"\blong[\s-]?term\b", ql):
            ambiguities.append(Ambiguity(field="horizon_years", message="'Long term' was mentioned without a number of years; no horizon-based checks were applied.", severity="warning"))
        elif re.search(r"\bshort[\s-]?term\b", ql):
            ambiguities.append(Ambiguity(field="horizon_years", message="'Short term' was mentioned without a number of years; no horizon-based checks were applied.", severity="warning"))
        return None

    @staticmethod
    def _extract_risk(ql: str, ambiguities: List[Ambiguity]) -> Optional[str]:
        # Negated phrasing: "don't want very high volatility/risk" => ceiling below Very High
        if re.search(r"(?:avoid|don'?t want|do not want|without|no)\s+very[\s-]high\s+(?:risk|volatil)", ql):
            ambiguities.append(Ambiguity(field="risk_tolerance", message="'No very high volatility' interpreted as a SEBI Risk-o-meter ceiling of 'High'.", severity="warning"))
            return "High"
        risk_match = re.search(
            r"\b(very high|moderately high|low to moderate|aggressive|high|moderate|medium|balanced|conservative|low|minimal|zero|no)\b[\s-]*(?:risk|volatility)",
            ql,
        )
        if not risk_match:
            risk_match = re.search(r"\brisk\s+(?:tolerance|appetite|profile)\s*(?:is|of|:)?\s*\b(very high|moderately high|low to moderate|aggressive|high|moderate|medium|low)\b", ql)
        if not risk_match:
            return None
        phrase = risk_match.group(1)
        label, exact = RISK_VOCABULARY[phrase]
        if not exact:
            ambiguities.append(
                Ambiguity(
                    field="risk_tolerance",
                    message=f"'{phrase} risk' is not a SEBI Risk-o-meter label; interpreted as a ceiling of '{label}' (3-tier to 6-level mapping documented in docs/decision_engine.md).",
                    severity="warning",
                    candidates=[lvl for lvl in RISK_LEVEL_ORDER if RISK_LEVEL_ORDER[lvl] <= RISK_LEVEL_ORDER[label]],
                )
            )
        return label

    @staticmethod
    def _extract_categories(ql: str) -> Tuple[List[str], List[str]]:
        subs = [name for pattern, name in SUB_CATEGORY_KEYWORDS.items() if re.search(pattern, ql)]
        cats = [name for pattern, name in CATEGORY_KEYWORDS.items() if re.search(pattern, ql)]
        # 'short term' alone should not trigger the Short Duration sub-category unless debt wording present
        return sorted(set(cats)), sorted(set(subs))

    @staticmethod
    def _extract_expense(ql: str) -> Optional[float]:
        # Number-first form ('0.3% expense ratio') is unambiguous, so it is preferred; the
        # keyword-first form may skip over unrelated percentages (e.g. '30% return').
        for pattern in (EXPENSE_RE_ALT, EXPENSE_RE):
            for match in pattern.finditer(ql):
                value = float(match.group(1))
                if 0 < value <= 5:
                    return value
        return None

    @staticmethod
    def _extract_min_return(ql: str, ambiguities: List[Ambiguity]) -> Optional[float]:
        match = MIN_RETURN_RE.search(ql)
        if not match:
            return None
        value = float(match.group(1))
        if value >= 25:
            ambiguities.append(Ambiguity(field="min_return_3y", message=f"A minimum return of {value:g}% per year was requested; this is evaluated against historical 3-year CAGR only and is not a forecast.", severity="warning"))
        return value / 100.0

    @staticmethod
    def _extract_min_aum(ql: str) -> Optional[float]:
        match = MIN_AUM_RE.search(ql)
        if not match:
            return None
        value = float(match.group(1).replace(",", ""))
        return value

    @staticmethod
    def _extract_preferences(ql: str, max_expense: Optional[float]) -> UserPreferences:
        weights: Dict[str, float] = {}
        for key, pattern in PREFERENCE_PATTERNS.items():
            if re.search(pattern, ql):
                weights[key] = 1.0
        if max_expense is not None:
            weights.setdefault("low_cost", 0.5)
        if re.search(r"\bmost important\b|\bpriorit(?:y|ise|ize)\b|\bcare more about\b", ql):
            for key in list(weights):
                weights[key] = 1.0
        return UserPreferences(weights=weights)

    @staticmethod
    def _extract_objective(ql: str) -> Optional[str]:
        for key, pattern in OBJECTIVE_PATTERNS.items():
            if re.search(pattern, ql):
                return key
        return None

    @staticmethod
    def _extract_plan(ql: str) -> Optional[str]:
        match = PLAN_RE.search(ql)
        return match.group(1).capitalize() if match else None

    @staticmethod
    def _extract_amcs(ql: str) -> Tuple[List[str], List[str]]:
        def _amc_in(phrase: str) -> Optional[str]:
            toks = _tokens(phrase)
            for t in toks:
                if t in AMC_TOKENS:
                    return t
            return None

        required: List[str] = []
        preferred: List[str] = []
        for match in ONLY_AMC_RE.finditer(ql):
            amc = _amc_in(match.group(1))
            if amc:
                required.append(amc)
        for match in PREFER_AMC_RE.finditer(ql):
            amc = _amc_in(match.group(1))
            if amc and amc not in required:
                preferred.append(amc)
        return required, preferred

    def _extract_targets(self, q: str, ql: str, ambiguities: List[Ambiguity]) -> Tuple[List[ComparisonTarget], str]:
        comparison_words = bool(re.search(r"\bcompare\b|\bcomparison\b|\bvs\.?\b|\bversus\b|\bdifference between\b", ql))
        explanation_words = bool(re.search(r"\bwhy\b|\bexplain\b|\breason\b", ql))
        filter_words = bool(re.search(r"\bwhich funds\b|\blist\b|\bfilter\b|\bshow me\b|\bfunds that\b", ql))

        targets: List[ComparisonTarget] = self.resolver.find_all(q)

        match = COMPARE_RE.search(q)
        if match:
            for mention in (match.group(1), match.group(2)):
                mention_clean = mention.strip(" ,.")
                if not any(t.fund_id and t.fund_name and _fund_key_tokens(t.fund_name)[0] in _tokens(mention_clean) for t in targets):
                    resolved, candidates = self.resolver.resolve(mention_clean)
                    if resolved:
                        targets.append(resolved)
                    elif candidates:
                        ambiguities.append(Ambiguity(field="comparison_targets", message=f"'{mention_clean}' matches several funds.", severity="blocking", candidates=candidates))
                    elif any(t in AMC_TOKENS for t in _tokens(mention_clean)):
                        ambiguities.append(Ambiguity(field="comparison_targets", message=f"Could not find a fund matching '{mention_clean}' in the database.", severity="blocking"))

        # de-duplicate by fund_id preserving order
        seen: set = set()
        unique: List[ComparisonTarget] = []
        for t in targets:
            key = t.fund_id or t.mention
            if key not in seen:
                seen.add(key)
                unique.append(t)

        if explanation_words and len(unique) >= 1:
            intent = "explanation"
        elif comparison_words or len(unique) >= 2:
            intent = "comparison"
        elif filter_words:
            intent = "filter"
        else:
            intent = "recommendation"

        if intent == "comparison" and len(unique) == 1:
            ambiguities.append(Ambiguity(field="comparison_targets", message=f"Only one fund ('{unique[0].fund_name}') could be identified for comparison; it will be compared against other eligible candidates.", severity="warning"))
        return unique, intent
