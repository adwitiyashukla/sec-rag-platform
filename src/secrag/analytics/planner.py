"""Numeric query planning.

Turns a numeric question into an explicit, executable plan against the XBRL
fact table.

This layer is rule-based on purpose, and the choice is worth defending. The
router that decides a question *is* numeric is a learned model, because intent
is fuzzy and phrasing varies without limit. But once that decision is made,
mapping "gross margin" to gross_profit divided by revenue is a definition, not
a prediction. Learning it would add a failure mode to something that has an
exact answer, and the resulting figures are meant to be auditable.

If the plan cannot be resolved with confidence, it returns None and the query
falls back to ordinary retrieval. Refusing to guess is the point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from secrag.core.logging import get_logger
from secrag.core.types import NumericResult
from secrag.ingest.xbrl import FactStore

log = get_logger(__name__)

Operation = Literal["value", "growth", "cagr", "ratio", "series"]

_YEAR_RE = re.compile(r"\b(?:fy\s?)?((?:19|20)\d{2})\b", re.IGNORECASE)

# Longest phrases first, so "operating cash flow" is not shadowed by "cash".
_METRIC_PHRASES: tuple[tuple[str, str], ...] = (
    ("operating cash flow", "operating_cash_flow"),
    ("cash from operations", "operating_cash_flow"),
    ("research and development", "rnd_expense"),
    ("r and d", "rnd_expense"),
    ("diluted earnings per share", "eps_diluted"),
    ("earnings per share", "eps_diluted"),
    ("diluted shares", "shares_diluted"),
    ("shares outstanding", "shares_diluted"),
    ("stockholders equity", "stockholders_equity"),
    ("shareholders equity", "stockholders_equity"),
    ("total liabilities", "total_liabilities"),
    ("total assets", "total_assets"),
    ("operating income", "operating_income"),
    ("gross profit", "gross_profit"),
    ("net income", "net_income"),
    ("net profit", "net_income"),
    ("revenue", "revenue"),
    ("sales", "revenue"),
    ("cash", "cash"),
    ("assets", "total_assets"),
    ("liabilities", "total_liabilities"),
    ("equity", "stockholders_equity"),
)

# Ratios are defined here rather than inferred, with their display label.
_RATIOS: tuple[tuple[str, str, str, str], ...] = (
    ("gross margin", "gross_profit", "revenue", "Gross margin"),
    ("operating margin", "operating_income", "revenue", "Operating margin"),
    ("net margin", "net_income", "revenue", "Net margin"),
    ("profit margin", "net_income", "revenue", "Net margin"),
    ("return on equity", "net_income", "stockholders_equity", "Return on equity"),
    ("roe", "net_income", "stockholders_equity", "Return on equity"),
    ("return on assets", "net_income", "total_assets", "Return on assets"),
    ("roa", "net_income", "total_assets", "Return on assets"),
)

# Matched as substrings, so "grow" also covers "growth" and "growing".
# Omitting the bare stem sent "how much did revenue grow from 2022 to 2024"
# down the single-value path and silently answered a different question.
_GROWTH_WORDS = (
    "grow",
    "grew",
    "increase",
    "decrease",
    "change",
    "changed",
    "rise",
    "rose",
    "fall",
    "fell",
    "decline",
    "up from",
    "down from",
)
_CAGR_WORDS = ("cagr", "compound annual", "compounded")
_SERIES_WORDS = ("trend", "over time", "each year", "history", "series", "year by year")


@dataclass(slots=True)
class NumericPlan:
    tickers: list[str]
    metric: str
    operation: Operation
    years: list[int] = field(default_factory=list)
    ratio_numerator: str = ""
    ratio_denominator: str = ""
    label: str = ""

    def describe(self) -> str:
        who = ", ".join(self.tickers)
        span_text = "-".join(str(y) for y in self.years) if self.years else "latest"
        return f"{self.operation}({self.metric}) for {who} over {span_text}"


def _find_tickers(question: str, store: FactStore) -> list[str]:
    """Match tickers by symbol or by company name, restricted to what is indexed."""
    known = store.tickers()
    if not known:
        return []
    found: list[str] = []
    upper = question.upper()

    for ticker in known:
        if re.search(rf"\b{re.escape(ticker)}\b", upper):
            found.append(ticker)
            continue
        rows = store.df[store.df["ticker"] == ticker]
        if rows.empty:
            continue
        # "Apple Inc." should match "apple", so compare on the leading token.
        company = str(rows.iloc[0]["company"])
        lead = re.split(r"[ ,.]", company.strip())[0]
        if len(lead) > 2 and re.search(rf"\b{re.escape(lead.upper())}\b", upper):
            found.append(ticker)

    return list(dict.fromkeys(found))


def _find_metric(lowered: str) -> str | None:
    for phrase, metric in _METRIC_PHRASES:
        if phrase in lowered:
            return metric
    return None


def _find_ratio(lowered: str) -> tuple[str, str, str] | None:
    for phrase, numerator, denominator, label in _RATIOS:
        if phrase in lowered:
            return numerator, denominator, label
    return None


def plan_numeric(question: str, store: FactStore) -> NumericPlan | None:
    """Build an executable numeric plan, or None if the question is not resolvable."""
    if store.is_empty:
        return None

    lowered = question.lower()
    tickers = _find_tickers(question, store)
    if not tickers:
        return None

    years = sorted({int(y) for y in _YEAR_RE.findall(question)})
    # Years outside the indexed range are almost always a misparse.
    available = set(store.years(tickers[0]))
    years = [y for y in years if y in available] or years

    if ratio := _find_ratio(lowered):
        numerator, denominator, label = ratio
        target_year = years[-1] if years else (store.latest_year(tickers[0], "revenue") or 0)
        return NumericPlan(
            tickers=tickers,
            metric=numerator,
            operation="ratio",
            years=[target_year],
            ratio_numerator=numerator,
            ratio_denominator=denominator,
            label=label,
        )

    metric = _find_metric(lowered)
    if metric is None:
        return None

    if any(word in lowered for word in _CAGR_WORDS) and len(years) >= 2:
        operation: Operation = "cagr"
    elif any(word in lowered for word in _GROWTH_WORDS) and len(years) >= 2:
        operation = "growth"
    elif any(word in lowered for word in _SERIES_WORDS):
        operation = "series"
    elif any(word in lowered for word in _GROWTH_WORDS) and len(years) == 1:
        # "How much did revenue grow in 2024" means 2023 to 2024.
        operation = "growth"
        years = [years[0] - 1, years[0]]
    else:
        operation = "value"

    if not years:
        latest = store.latest_year(tickers[0], metric)
        if latest is None:
            return None
        years = [latest]

    return NumericPlan(tickers=tickers, metric=metric, operation=operation, years=years)


def execute_plan(plan: NumericPlan, store: FactStore) -> list[NumericResult]:
    """Run a plan against the fact table."""
    results: list[NumericResult] = []

    for ticker in plan.tickers:
        match plan.operation:
            case "ratio":
                results.append(
                    store.ratio(
                        ticker,
                        plan.ratio_numerator,
                        plan.ratio_denominator,
                        plan.years[-1],
                        plan.label,
                    )
                )
            case "growth":
                results.append(store.growth(ticker, plan.metric, plan.years[0], plan.years[-1]))
            case "cagr":
                results.append(store.cagr(ticker, plan.metric, plan.years[0], plan.years[-1]))
            case "series":
                frame = store.series(ticker, plan.metric)
                for _, row in frame.iterrows():
                    results.append(store.value_of(ticker, plan.metric, int(row["fiscal_year"])))
            case _:
                for year in plan.years:
                    results.append(store.value_of(ticker, plan.metric, year))

    resolved = [r for r in results if r.value is not None]
    log.info(
        "numeric_plan_executed",
        plan=plan.describe(),
        resolved=len(resolved),
        total=len(results),
    )
    return resolved
