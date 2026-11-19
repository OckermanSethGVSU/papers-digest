"""Daily vector-search paper screen: fetch -> triage with Claude -> digest -> write."""

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic

from consts import (
    ARXIV_API,
    ARXIV_CATEGORIES,
    ARXIV_NS,
    ARXIV_PAGE_SIZE,
    ARXIV_REQUEST_DELAY_S,
    BATCH_POLL_INTERVAL_S,
    BATCH_TIMEOUT_S,
    DIGEST_EFFORT,
    DIGEST_MAX_TOKENS,
    DIGEST_MODEL,
    DIGEST_OUTPUT_SCHEMA,
    FALLBACK_BETA,
    FOCUS_DESCRIPTION,
    HF_LOOKBACK_DAYS,
    HF_PAPERS_API,
    LOOKBACK_DAYS,
    MAX_ARXIV_RESULTS_PER_CATEGORY,
    MAX_PAPERS_IN_DIGEST,
    MIN_TRIAGE_SCORE,
    MODELS_WITHOUT_FALLBACK,
    SEEN_RETENTION_DAYS,
    TRIAGE_BATCH_SIZE,
    TRIAGE_EFFORT,
    TRIAGE_MAX_TOKENS,
    TRIAGE_MODEL,
    TRIAGE_OUTPUT_SCHEMA,
    USE_BATCH_API,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEN_PATH = REPO_ROOT / "digests" / "seen.json"
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


# --- fetching ------------------------------------------------------------------

def arxiv_id_from(url: str) -> str:
    """Return the version-less arXiv id in a URL, or '' if none."""
    match = ARXIV_ID_RE.search(url)
    return match.group(1) if match else ""


def paper_key(paper: dict) -> str:
    """Stable identity: arXiv id when available, else normalized title."""
    return paper["arxiv_id"] or "title:" + re.sub(r"\W+", "", paper["title"].lower())


def http_get(url: str, attempts: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "paper-digest/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_arxiv_papers(category: str, start_date: str, end_date: str) -> list[dict]:
    """
    fetch all papers in an arxiv category submitted within the date range, paginated.

    Args:
        - category: e.g. "cs.IR"
        - start_date, end_date: YYYYMMDDHHMM
    """
    papers = []
    start = 0
    while start < MAX_ARXIV_RESULTS_PER_CATEGORY:
        params = urllib.parse.urlencode({
            "search_query": f"cat:{category} AND submittedDate:[{start_date} TO {end_date}]",
            "start": start,
            "max_results": ARXIV_PAGE_SIZE,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        root = ET.fromstring(http_get(f"{ARXIV_API}?{params}"))
        entries = root.findall("atom:entry", ARXIV_NS)
        for entry in entries:
            link = entry.findtext("atom:id", "", ARXIV_NS).strip()
            papers.append({
                "title": re.sub(r"\s+", " ", entry.findtext("atom:title", "", ARXIV_NS).strip()),
                "abstract": re.sub(r"\s+", " ", entry.findtext("atom:summary", "", ARXIV_NS).strip()),
                "authors": [a.findtext("atom:name", "", ARXIV_NS) for a in entry.findall("atom:author", ARXIV_NS)],
                "url": re.sub(r"v\d+$", "", link),
                "arxiv_id": arxiv_id_from(link),
                "published": entry.findtext("atom:published", "", ARXIV_NS)[:10],
                "source": f"arXiv ({category})",
            })
        start += ARXIV_PAGE_SIZE
        time.sleep(ARXIV_REQUEST_DELAY_S)
        if len(entries) < ARXIV_PAGE_SIZE:
            break
    return papers


def fetch_hf_papers(date_str: str) -> list[dict]:
    """
    fetch papers from huggingface daily papers for a given date (YYYY-MM-DD).
    """
    try:
        data = json.loads(http_get(f"{HF_PAPERS_API}?date={date_str}"))
    except Exception as error:
        print(f"  HuggingFace fetch failed for {date_str}: {error}")
        return []

    papers = []
    for item in data:
        paper = item.get("paper", {})
        arxiv_id = paper.get("id", "")
        papers.append({
            "title": paper.get("title", "").strip(),
            "abstract": paper.get("summary", "").strip(),
            "authors": [a.get("name", "") for a in paper.get("authors", [])],
            "url": f"https://huggingface.co/papers/{arxiv_id}" if arxiv_id else "",
            "arxiv_id": arxiv_id_from(arxiv_id),
            "published": paper.get("publishedAt", "")[:10],
            "source": "HuggingFace",
        })
    return papers


def deduplicate(papers: list[dict]) -> list[dict]:
    """Drop duplicates by arXiv id / title. Earlier entries (arXiv) win over later (HF)."""
    seen = set()
    unique = []
    for p in papers:
        key = paper_key(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def collect_papers(now: datetime) -> list[dict]:
    """Gather papers from all sources and deduplicate."""
    start_str = (now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d0000")
    end_str = now.strftime("%Y%m%d2359")

    all_papers = []
    for cat in ARXIV_CATEGORIES:
        print(f"fetching arXiv {cat}...")
        all_papers.extend(fetch_arxiv_papers(cat, start_str, end_str))

    for day_offset in range(HF_LOOKBACK_DAYS):
        date = (now - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        print(f"fetching HuggingFace papers for {date}...")
        all_papers.extend(fetch_hf_papers(date))

    print(f"total fetched: {len(all_papers)}")
    unique = deduplicate(all_papers)
    print(f"after dedup: {len(unique)}")
    return unique


# --- seen-paper ledger -------------------------------------------------------------

def load_seen(now: datetime) -> dict[str, str]:
    """Load {paper_key: first_seen_date}, pruning entries older than the retention window."""
    if not SEEN_PATH.exists():
        return {}
    seen = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
    cutoff = (now - timedelta(days=SEEN_RETENTION_DAYS)).strftime("%Y-%m-%d")
    return {k: v for k, v in seen.items() if v >= cutoff}


def save_seen(seen: dict[str, str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.write_text(json.dumps(dict(sorted(seen.items())), indent=0) + "\n", encoding="utf-8")


# --- claude ----------------------------------------------------------------------

DIRECT_CONCURRENCY = 4


def parse_message(message) -> dict:
    """Return the structured JSON from a response, or raise if it is unusable."""
    if message.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined the request ({message.stop_details}).")
    if message.stop_reason == "max_tokens":
        raise RuntimeError("Claude hit max_tokens before finishing the JSON output.")
    text = next(block.text for block in message.content if block.type == "text")
    return json.loads(text)


def build_params(*, model: str, effort: str, max_tokens: int, system: str, prompt: str, schema: dict) -> dict:
    """Request parameters shared by the batch and direct paths."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }


def run_direct(client: anthropic.Anthropic, params: dict) -> dict:
    """Full-price streaming call, with the server-side refusal fallback where supported."""
    if params["model"] in MODELS_WITHOUT_FALLBACK:
        with client.messages.stream(**params) as stream:
            return parse_message(stream.get_final_message())
    with client.beta.messages.stream(**params, betas=[FALLBACK_BETA], fallbacks="default") as stream:
        return parse_message(stream.get_final_message())


def run_batch(client: anthropic.Anthropic, requests: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    """Submit requests to the Batches API. Returns (parsed results, ids that need a direct retry)."""
    batch = client.messages.batches.create(
        requests=[{"custom_id": custom_id, "params": params} for custom_id, params in requests.items()]
    )
    print(f"  submitted batch {batch.id} ({len(requests)} requests)")
    deadline = time.monotonic() + BATCH_TIMEOUT_S
    while batch.processing_status != "ended":
        if time.monotonic() > deadline:
            print("  batch timed out; canceling and finishing the rest directly")
            client.messages.batches.cancel(batch.id)
            while batch.processing_status != "ended":
                time.sleep(BATCH_POLL_INTERVAL_S)
                batch = client.messages.batches.retrieve(batch.id)
            break
        time.sleep(BATCH_POLL_INTERVAL_S)
        batch = client.messages.batches.retrieve(batch.id)

    results, retry = {}, []
    for entry in client.messages.batches.results(batch.id):
        if entry.result.type == "succeeded":
            try:
                results[entry.custom_id] = parse_message(entry.result.message)
                continue
            except (RuntimeError, json.JSONDecodeError, StopIteration) as error:
                print(f"  {entry.custom_id}: {error}")
        else:
            print(f"  {entry.custom_id}: batch result {entry.result.type}")
        retry.append(entry.custom_id)
    retry += [custom_id for custom_id in requests if custom_id not in results and custom_id not in retry]
    return results, retry


def run_requests(client: anthropic.Anthropic, requests: dict[str, dict]) -> dict[str, dict]:
    """Run every request, via the Batches API when enabled, retrying failures directly."""
    if USE_BATCH_API:
        results, retry = run_batch(client, requests)
    else:
        results, retry = {}, list(requests)
    if retry:
        if USE_BATCH_API:
            print(f"  retrying {len(retry)} request(s) directly")
        with ThreadPoolExecutor(max_workers=DIRECT_CONCURRENCY) as pool:
            for custom_id, parsed in zip(retry, pool.map(lambda c: run_direct(client, requests[c]), retry)):
                results[custom_id] = parsed
    return results


def triage_params(batch: list[dict]) -> dict:
    items = [{"id": i, "title": p["title"], "abstract": p["abstract"]} for i, p in enumerate(batch)]
    prompt = (
        f"Score each of these {len(items)} papers for relevance to the reader described in the "
        "system prompt. Return a score for every id.\n\n"
        + json.dumps(items, ensure_ascii=False)
    )
    system = (
        "You screen new research papers for a vector search engineer.\n\n"
        f"{FOCUS_DESCRIPTION}\n\n"
        "Scores: 3 = core vector search; 2 = technique plausibly transfers to vector "
        "indexing, compression or search even if the paper never mentions it; 1 = tangential; "
        "0 = irrelevant. Most papers are 0. Judge from what the method does, not from keywords."
    )
    return build_params(
        model=TRIAGE_MODEL, effort=TRIAGE_EFFORT, max_tokens=TRIAGE_MAX_TOKENS,
        system=system, prompt=prompt, schema=TRIAGE_OUTPUT_SCHEMA,
    )


def scores_from(result: dict, batch_len: int) -> dict[int, int]:
    return {item["id"]: item["score"] for item in result["papers"] if 0 <= item["id"] < batch_len}


def triage_batch(client: anthropic.Anthropic, batch: list[dict]) -> dict[int, int]:
    """Score one batch directly (no Batches API). Used by check_triage.py."""
    return scores_from(run_direct(client, triage_params(batch)), len(batch))


def triage(client: anthropic.Anthropic, papers: list[dict]) -> list[dict]:
    """Score every paper and return those at or above MIN_TRIAGE_SCORE, best first."""
    batches = [papers[i:i + TRIAGE_BATCH_SIZE] for i in range(0, len(papers), TRIAGE_BATCH_SIZE)]
    print(f"triaging {len(papers)} papers in {len(batches)} requests...")
    results = run_requests(client, {f"triage-{i}": triage_params(b) for i, b in enumerate(batches)})

    scored = []
    for i, batch in enumerate(batches):
        scores = scores_from(results[f"triage-{i}"], len(batch))
        for index, paper in enumerate(batch):
            score = scores.get(index, 0)
            if score >= MIN_TRIAGE_SCORE:
                scored.append({**paper, "triage_score": score})
    scored.sort(key=lambda p: p["triage_score"], reverse=True)
    print(f"passed triage: {len(scored)}")
    if len(scored) > MAX_PAPERS_IN_DIGEST:
        scored = scored[:MAX_PAPERS_IN_DIGEST]
        print(f"capped at {MAX_PAPERS_IN_DIGEST}")
    return scored


def summarize(client: anthropic.Anthropic, papers: list[dict]) -> dict:
    """Cluster and summarize the papers that passed triage."""
    papers_for_prompt = [
        {"title": p["title"], "url": p["url"], "authors": p["authors"][:5],
         "published": p["published"], "source": p["source"], "abstract": p["abstract"]}
        for p in papers
    ]
    system = (
        "You write a daily research digest for a vector search engineer.\n\n"
        f"{FOCUS_DESCRIPTION}"
    )
    prompt = f"""These {len(papers_for_prompt)} papers passed a relevance screen today. Produce the digest.

- Pick 0-2 highlights worth a deep read (none if nothing stands out).
- Group the papers into 1-5 thematic clusters; use fewer clusters when there are few papers.
- Score relevance 1-3 (3 = core vector search, 2 = transferable technique, 1 = tangential).
- For each paper, write a 2-3 sentence summary and one sentence on its vector search angle:
  what an index/quantization/systems engineer could take from it. Be concrete.
- Drop papers that turn out not to be relevant after reading the abstract.

Papers:
{json.dumps(papers_for_prompt, indent=2, ensure_ascii=False)}"""
    params = build_params(
        model=DIGEST_MODEL, effort=DIGEST_EFFORT, max_tokens=DIGEST_MAX_TOKENS,
        system=system, prompt=prompt, schema=DIGEST_OUTPUT_SCHEMA,
    )
    return run_requests(client, {"digest": params})["digest"]


# --- output ----------------------------------------------------------------------

def render_markdown(digest: dict, date_str: str) -> str:
    """Render the digest JSON as Markdown."""
    lines = [f"# Vector Search Digest — {date_str}", ""]

    if digest.get("highlights"):
        lines += ["## Highlights", ""]
        for h in digest["highlights"]:
            lines.append(f"- **[{h['title']}]({h['url']})** — {h['reason']}")
        lines.append("")

    for cluster in digest.get("clusters", []):
        lines += [f"## {cluster['theme']}", ""]
        for p in cluster.get("papers", []):
            stars = p.get("relevance", 1) * "⭐"
            authors = p.get("authors", [])
            author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            lines.append(f"### [{p['title']}]({p['url']})")
            lines.append(f"_{author_str} | {p.get('published', '')} | {p.get('source', '')} | {stars}_")
            lines.append("")
            lines.append(p.get("summary", ""))
            if p.get("vector_search_angle"):
                lines += ["", f"**Vector search angle:** {p['vector_search_angle']}"]
            lines.append("")
        lines += ["---", ""]

    return "\n".join(lines)


def digest_path_for(now: datetime) -> Path:
    return REPO_ROOT / "digests" / "daily" / now.strftime("%Y") / now.strftime("%m") / f"{now:%Y-%m-%d}.md"


def set_github_output(name: str, value: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def main():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    set_github_output("digest_path", "")

    seen = load_seen(now)
    papers = [p for p in collect_papers(now) if paper_key(p) not in seen]
    print(f"new (not seen before): {len(papers)}")
    if not papers:
        save_seen(seen)
        print("no new papers today.")
        return

    client = anthropic.Anthropic()
    relevant = triage(client, papers)

    # Mark everything as seen only after triage succeeded, so a failed run retries tomorrow.
    for p in papers:
        seen[paper_key(p)] = today
    if not relevant:
        save_seen(seen)
        print("no relevant papers today, skipping digest.")
        return

    print(f"\nsummarizing {len(relevant)} papers...")
    digest = summarize(client, relevant)
    if not digest.get("clusters"):
        save_seen(seen)
        print("digest came back empty, skipping.")
        return

    path = digest_path_for(now)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(digest, today), encoding="utf-8")
    save_seen(seen)
    set_github_output("digest_path", str(path.relative_to(REPO_ROOT)))
    print(f"\ndigest written to {path}")


if __name__ == "__main__":
    main()
