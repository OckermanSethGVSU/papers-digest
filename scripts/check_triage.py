"""Sanity-check the triage prompt on known papers.

Fetches the given arXiv ids, runs them through the same triage call the daily job uses,
and prints each score. Exits non-zero if an expected-relevant paper scores below
MIN_TRIAGE_SCORE or an expected-irrelevant paper scores at or above it.

    python scripts/check_triage.py                      # built-in examples
    python scripts/check_triage.py --relevant 2611.08306 --irrelevant 2611.00001
"""

import argparse
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET

import anthropic

from consts import ARXIV_API, ARXIV_NS, MIN_TRIAGE_SCORE
from digest import arxiv_id_from, http_get, triage_batch

# Papers the screen must flag. Add to this when it misses something you cared about.
EXPECTED_RELEVANT = [
    "2611.08306",  # GINGER: graph-based vector search library
    "2611.09501",  # e-means: faster k-means (indirect: IVF / PQ training)
]
# Papers the screen should skip. Add false positives here.
EXPECTED_IRRELEVANT: list[str] = []


def fetch_by_ids(ids: list[str]) -> list[dict]:
    params = urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    root = ET.fromstring(http_get(f"{ARXIV_API}?{params}"))
    papers = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        papers.append({
            "arxiv_id": arxiv_id_from(entry.findtext("atom:id", "", ARXIV_NS)),
            "title": re.sub(r"\s+", " ", entry.findtext("atom:title", "", ARXIV_NS).strip()),
            "abstract": re.sub(r"\s+", " ", entry.findtext("atom:summary", "", ARXIV_NS).strip()),
        })
    return papers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--relevant", nargs="*", default=EXPECTED_RELEVANT)
    parser.add_argument("--irrelevant", nargs="*", default=EXPECTED_IRRELEVANT)
    args = parser.parse_args()

    expected = {i: True for i in args.relevant} | {i: False for i in args.irrelevant}
    papers = fetch_by_ids(list(expected))
    scores = triage_batch(anthropic.Anthropic(), papers)

    failures = 0
    for index, paper in enumerate(papers):
        score = scores.get(index, 0)
        want_relevant = expected[paper["arxiv_id"]]
        ok = (score >= MIN_TRIAGE_SCORE) == want_relevant
        failures += not ok
        label = "relevant" if want_relevant else "irrelevant"
        print(f"{'PASS' if ok else 'FAIL'}  score={score}  expected {label:10}  {paper['arxiv_id']}  {paper['title']}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
