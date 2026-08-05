"""XBRL structured financial facts.

This is the piece that stops the system inventing numbers.

Language models are unreliable at reading figures out of 10-K tables. The
tables are large, deeply nested, span several periods, and use scaling
footnotes ("in millions, except per share data") that sit far away from the
figures they govern. A model asked for revenue growth will usually produce
something plausible and wrong.

The SEC publishes the same figures as structured XBRL data. So numeric
questions are not answered from prose at all: the fact is looked up, the
arithmetic is done in pandas, and the result carries its formula and inputs so
a user can audit it without trusting the model. Retrieval still supplies the
narrative context around the number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from secrag.core.config import Settings, get_settings
from secrag.core.errors import IngestionError
from secrag.core.logging import get_logger
from secrag.core.types import NumericResult
from secrag.observability.tracing import span

log = get_logger(__name__)

# Companies tag the same economic concept differently, and the tag they use can
# change between years. Each logical metric therefore maps to an ordered list of
# US-GAAP tags, tried most-specific first.
CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "stockholders_equity": ("StockholdersEquity",),
    "cash": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
    "rnd_expense": ("ResearchAndDevelopmentExpense",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "shares_diluted": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
}

METRIC_LABELS = {
    "revenue": "Revenue",
    "net_income": "Net income",
    "gross_profit": "Gross profit",
    "operating_income": "Operating income",
    "total_assets": "Total assets",
    "total_liabilities": "Total liabilities",
    "stockholders_equity": "Stockholders equity",
    "cash": "Cash and equivalents",
    "rnd_expense": "Research and development expense",
    "operating_cash_flow": "Operating cash flow",
    "eps_diluted": "Diluted EPS",
    "shares_diluted": "Diluted shares outstanding",
}


# An annual period is nominally 365 days, but fiscal calendars use 52 and 53
# week years and companies occasionally file transition periods. This window
# accepts genuine annual periods while excluding quarters, which is the
# distinction that actually matters.
_MIN_ANNUAL_DAYS = 340
_MAX_ANNUAL_DAYS = 400


def _parse_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _annual_period(obs: dict[str, Any]) -> tuple[int, int] | None:
    """Return (fiscal_year, duration_days) if this observation is annual.

    Fiscal year is taken from the calendar year the period *ends* in. That is
    the convention US filers actually use, verified against real data rather
    than reasoned about: NVIDIA's year ending 26 January 2025 is its fiscal
    2025, and Walmart's year ending 31 January 2026 is its fiscal 2026.

    An earlier version labelled by the period midpoint, on the theory that a
    February-to-January year belongs to the calendar year holding most of it.
    That is defensible in the abstract and wrong in practice: it shifted every
    January-year-end filer back by one, so NVIDIA's 130.5 billion fiscal 2025
    was filed under FY2024 and silently disagreed with the chunk metadata,
    which takes its year from the filing's own report date.

    Balance sheet concepts are instants with no start date. They are dated by
    their measurement date, which is the fiscal year end, so they are accepted
    with a duration of zero.
    """
    end = _parse_date(obs.get("end"))
    if end is None:
        return None

    start = _parse_date(obs.get("start"))
    if start is None:
        days = 0
    else:
        days = (end - start).days
        if not (_MIN_ANNUAL_DAYS <= days <= _MAX_ANNUAL_DAYS):
            return None

    return end.year, days


@dataclass(frozen=True, slots=True)
class Fact:
    ticker: str
    company: str
    metric: str
    concept: str
    fiscal_year: int
    value: float
    unit: str
    form: str
    frame: str
    accession: str


class FactStore:
    """A tidy table of company facts with a small analytics layer on top."""

    COLUMNS = (
        "ticker",
        "company",
        "metric",
        "concept",
        "fiscal_year",
        "value",
        "unit",
        "form",
        "frame",
        "accession",
    )

    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        self.df = frame if frame is not None else pd.DataFrame(columns=list(self.COLUMNS))

    # -- construction -----------------------------------------------------

    @classmethod
    def from_company_facts(
        cls, payload: dict[str, Any], ticker: str, existing: pd.DataFrame | None = None
    ) -> FactStore:
        """Flatten the EDGAR companyfacts document into tidy annual rows.

        Two details here are the difference between correct figures and
        confidently wrong ones:

        Period selection. Each observation carries "fp" and "fy", but those
        describe the *report* the fact appeared in, not the period the fact
        covers. A 10-K also tags its fourth-quarter figures, and those carry
        fp="FY" and form="10-K" too. Filtering on those fields alone silently
        admits quarterly values as annual ones. Apple's FY2020 revenue then
        reads as 64.7 billion, its Q4 number, instead of 274.5 billion, and
        every growth rate computed from it is wrong while looking entirely
        plausible. Annual facts are therefore selected by measuring the actual
        start-to-end duration.

        Fiscal year labelling. The year is taken from the period midpoint
        rather than its end date, so companies whose fiscal year closes in
        January are not shifted a year forward.
        """
        company = str(payload.get("entityName", ticker))
        us_gaap = payload.get("facts", {}).get("us-gaap", {})
        rows: list[dict[str, Any]] = []

        for metric, aliases in CONCEPT_ALIASES.items():
            # Alias preference is resolved per fiscal year, not per concept.
            #
            # Stopping at the first alias that yields any data at all looks
            # reasonable and silently loses years. NVIDIA tagged revenue as
            # RevenueFromContractWithCustomerExcludingAssessedTax through
            # fiscal 2022 and as Revenues from fiscal 2023 onward. Breaking on
            # the first alias found the old tag, stopped, and dropped every
            # recent year including the 130.5 billion fiscal 2025. Companies
            # change their tagging, so every alias is collected and the
            # highest-priority one available for each year wins.
            for priority, concept in enumerate(aliases):
                entry = us_gaap.get(concept)
                if not entry:
                    continue

                for unit, observations in entry.get("units", {}).items():
                    for obs in observations:
                        if not str(obs.get("form", "")).startswith("10-K"):
                            continue
                        if (value := obs.get("val")) is None:
                            continue

                        period = _annual_period(obs)
                        if period is None:
                            continue
                        fiscal_year, days = period

                        rows.append(
                            {
                                "ticker": ticker.upper(),
                                "company": company,
                                "metric": metric,
                                "concept": concept,
                                "fiscal_year": fiscal_year,
                                "value": float(value),
                                "unit": unit,
                                "form": str(obs.get("form", "10-K")),
                                "frame": str(obs.get("frame", "")),
                                "accession": str(obs.get("accn", "")),
                                "_filed": str(obs.get("filed", "")),
                                "_priority": priority,
                                "_days": days,
                            }
                        )

        frame = pd.DataFrame(rows)
        if not frame.empty:
            # Sorted so the winner per (ticker, metric, year) is the most
            # preferred alias, and within that the most recently filed value,
            # which is what picks up restatements.
            frame = (
                frame.sort_values(
                    ["_priority", "_filed", "accession"], ascending=[False, True, True]
                )
                .drop_duplicates(subset=["ticker", "metric", "fiscal_year"], keep="last")
                .drop(columns=["_filed", "_days", "_priority"])
                .sort_values(["ticker", "metric", "fiscal_year"])
                .reset_index(drop=True)
            )
        frame = frame.reindex(columns=list(cls.COLUMNS))

        if existing is not None and not existing.empty:
            frame = (
                pd.concat([existing, frame], ignore_index=True)
                .drop_duplicates(subset=["ticker", "metric", "fiscal_year"], keep="last")
                .reset_index(drop=True)
            )

        store = cls(frame)
        store._warn_on_implausible(ticker)
        return store

    def _warn_on_implausible(self, ticker: str) -> None:
        """Flag year-over-year jumps that suggest a period mismatch.

        A genuine annual revenue change above 200 percent is rare; a period
        selection bug produces them constantly. Logging it turns a silent data
        error into something visible during ingestion.
        """
        if self.df.empty:
            return
        for metric in ("revenue", "net_income", "total_assets"):
            series = self.series(ticker, metric)
            if series.empty or "yoy_pct" not in series:
                continue
            suspicious = series[series["yoy_pct"].abs() > 200]
            for _, row in suspicious.iterrows():
                log.warning(
                    "xbrl_implausible_change",
                    ticker=ticker,
                    metric=metric,
                    fiscal_year=int(row["fiscal_year"]),
                    yoy_pct=float(row["yoy_pct"]),
                )
        log.info("xbrl_facts_loaded", ticker=ticker, rows=len(self.df))

    # -- persistence ------------------------------------------------------

    def save(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        settings.ensure_dirs()
        self.df.to_parquet(settings.index_dir / "facts.parquet", index=False)

    @classmethod
    def load(cls, settings: Settings | None = None) -> FactStore:
        settings = settings or get_settings()
        path = settings.index_dir / "facts.parquet"
        if not path.exists():
            return cls()
        return cls(pd.read_parquet(path))

    # -- queries ----------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return self.df.empty

    def tickers(self) -> list[str]:
        return [] if self.is_empty else sorted(self.df["ticker"].unique().tolist())

    def years(self, ticker: str) -> list[int]:
        if self.is_empty:
            return []
        subset = self.df[self.df["ticker"] == ticker.upper()]
        return sorted(int(y) for y in subset["fiscal_year"].unique())

    def get(self, ticker: str, metric: str, fiscal_year: int) -> Fact | None:
        if self.is_empty:
            return None
        match = self.df[
            (self.df["ticker"] == ticker.upper())
            & (self.df["metric"] == metric)
            & (self.df["fiscal_year"] == fiscal_year)
        ]
        if match.empty:
            return None
        row = match.iloc[-1]
        return Fact(
            ticker=str(row["ticker"]),
            company=str(row["company"]),
            metric=str(row["metric"]),
            concept=str(row["concept"]),
            fiscal_year=int(row["fiscal_year"]),
            value=float(row["value"]),
            unit=str(row["unit"]),
            form=str(row["form"]),
            frame=str(row["frame"]),
            accession=str(row["accession"]),
        )

    def latest_year(self, ticker: str, metric: str) -> int | None:
        years = [
            int(y)
            for y in self.df[(self.df["ticker"] == ticker.upper()) & (self.df["metric"] == metric)][
                "fiscal_year"
            ].tolist()
        ]
        return max(years) if years else None

    # -- analytics --------------------------------------------------------

    def value_of(self, ticker: str, metric: str, fiscal_year: int) -> NumericResult:
        fact = self.get(ticker, metric, fiscal_year)
        label = f"{METRIC_LABELS.get(metric, metric)} for {ticker.upper()} FY{fiscal_year}"
        if fact is None:
            return NumericResult(label=label, value=None, formula="fact not reported")
        return NumericResult(
            label=label,
            value=fact.value,
            unit=fact.unit,
            formula=f"{fact.concept} as filed",
            inputs={f"FY{fiscal_year}": fact.value},
            period=f"FY{fiscal_year}",
            concept=fact.concept,
        )

    def growth(self, ticker: str, metric: str, start_year: int, end_year: int) -> NumericResult:
        """Percentage change between two fiscal years."""
        label = (
            f"{METRIC_LABELS.get(metric, metric)} growth for "
            f"{ticker.upper()} FY{start_year} to FY{end_year}"
        )
        start = self.get(ticker, metric, start_year)
        end = self.get(ticker, metric, end_year)
        if start is None or end is None or start.value == 0:
            return NumericResult(label=label, value=None, formula="insufficient data")
        change = (end.value - start.value) / abs(start.value) * 100.0
        return NumericResult(
            label=label,
            value=round(change, 2),
            unit="percent",
            formula=f"(FY{end_year} - FY{start_year}) / |FY{start_year}| x 100",
            inputs={f"FY{start_year}": start.value, f"FY{end_year}": end.value},
            period=f"FY{start_year}-FY{end_year}",
            concept=end.concept,
        )

    def cagr(self, ticker: str, metric: str, start_year: int, end_year: int) -> NumericResult:
        label = f"{METRIC_LABELS.get(metric, metric)} CAGR for {ticker.upper()}"
        start = self.get(ticker, metric, start_year)
        end = self.get(ticker, metric, end_year)
        periods = end_year - start_year
        if start is None or end is None or periods <= 0 or start.value <= 0 or end.value <= 0:
            return NumericResult(label=label, value=None, formula="insufficient data")
        value = ((end.value / start.value) ** (1 / periods) - 1) * 100.0
        return NumericResult(
            label=label,
            value=round(value, 2),
            unit="percent",
            formula=f"(FY{end_year} / FY{start_year}) ^ (1/{periods}) - 1",
            inputs={f"FY{start_year}": start.value, f"FY{end_year}": end.value},
            period=f"FY{start_year}-FY{end_year}",
            concept=end.concept,
        )

    def ratio(
        self, ticker: str, numerator: str, denominator: str, fiscal_year: int, label: str
    ) -> NumericResult:
        top = self.get(ticker, numerator, fiscal_year)
        bottom = self.get(ticker, denominator, fiscal_year)
        full_label = f"{label} for {ticker.upper()} FY{fiscal_year}"
        if top is None or bottom is None or bottom.value == 0:
            return NumericResult(label=full_label, value=None, formula="insufficient data")
        return NumericResult(
            label=full_label,
            value=round(top.value / bottom.value * 100.0, 2),
            unit="percent",
            formula=f"{numerator} / {denominator} x 100",
            inputs={numerator: top.value, denominator: bottom.value},
            period=f"FY{fiscal_year}",
            concept=top.concept,
        )

    def compare(self, tickers: Sequence[str], metric: str, fiscal_year: int) -> list[NumericResult]:
        return [self.value_of(t, metric, fiscal_year) for t in tickers]

    def series(self, ticker: str, metric: str) -> pd.DataFrame:
        """Full annual series for one metric, with year-over-year change."""
        if self.is_empty:
            return pd.DataFrame(columns=["fiscal_year", "value", "yoy_pct"])
        subset = (
            self.df[(self.df["ticker"] == ticker.upper()) & (self.df["metric"] == metric)]
            .sort_values("fiscal_year")
            .loc[:, ["fiscal_year", "value"]]
            .reset_index(drop=True)
        )
        if subset.empty:
            return pd.DataFrame(columns=["fiscal_year", "value", "yoy_pct"])
        subset["yoy_pct"] = (subset["value"].pct_change() * 100).round(2)
        return subset


async def build_fact_store(tickers: Sequence[str], settings: Settings | None = None) -> FactStore:
    """Fetch and flatten XBRL facts for each ticker."""
    from secrag.ingest.edgar import EdgarClient

    settings = settings or get_settings()

    # Seed from what is already on disk. Building from None means ingesting a
    # single new ticker silently deletes the facts for every other company.
    existing = FactStore.load(settings)
    frame: pd.DataFrame | None = None if existing.is_empty else existing.df

    async with EdgarClient(settings) as client:
        for ticker in tickers:
            try:
                with span("xbrl_fetch", ticker=ticker):
                    cik, _ = await client.resolve_ticker(ticker)
                    payload = await client.company_facts(cik)
                frame = FactStore.from_company_facts(payload, ticker, frame).df
            except IngestionError as exc:
                log.warning("xbrl_failed", ticker=ticker, error=exc.message)

    store = FactStore(frame)
    store.save(settings)
    return store
