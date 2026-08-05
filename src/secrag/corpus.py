"""The demo corpus definition.

Kept in one place so ingestion, evaluation, and documentation cannot disagree
about which filings the system is supposed to contain. A golden set that
references a company nobody ingested produces a silent zero rather than an
error, so this is a single source of truth on purpose.

The five companies are chosen to stress different parts of the pipeline: two
large technology filers with dense narrative sections, a semiconductor company
whose figures move sharply year over year, a bank whose financial statements use
an entirely different structure and whose 10-K is several times longer than the
others, and a retailer whose fiscal year ends on 31 January.

That last one is not decoration. A fiscal year running February to January must
be labelled by the calendar year holding most of it, and getting that wrong
shifts every Walmart figure forward by one year. Keeping it in the corpus means
the midpoint labelling in the XBRL loader is exercised on real data rather than
only in a unit test.

Exxon was the original fifth pick and had to be dropped, which is worth
recording. The ticker XOM now resolves through EDGAR's company index to
"ExxonMobil Holdings Corp" (CIK 0002115436), an entity created in a
reorganisation that has filed 8-Ks and 10-Qs but no 10-K. Ticker to CIK mapping
is not stable over time, and a pipeline that assumes it is will quietly return
nothing.
"""

from __future__ import annotations

CORPUS_TICKERS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "JPM", "WMT")
CORPUS_YEARS: int = 2

COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "WMT": "Walmart Inc.",
}
