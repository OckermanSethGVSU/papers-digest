# paper-digest

A daily screen of new arXiv and HuggingFace papers for anything relevant to **vector search**: ANN indexes, quantization, vector databases, and techniques that transfer to them even when a paper never says "vector search" (for example a faster k-means, which is how IVF partitions are built).

A GitHub Actions workflow runs every day at 07:00 UTC:

1. **Fetch** every paper submitted in the last 4 days to `cs.IR`, `cs.DB`, `cs.DS`, `cs.CG`, `cs.LG`, `cs.CL`, `cs.DC`, `cs.AR` and `cs.PF`, plus the last 2 days of HuggingFace daily papers. Papers already in `digests/seen.json` are skipped, so each paper is screened once.
2. **Triage**: Claude reads each title and abstract and scores it 0–3 against the focus description in `scripts/consts.py`. There is no keyword prefilter, so indirectly relevant papers get through.
3. **Digest**: the papers scoring 2 or higher (best 25) are clustered and summarized, each with a one-line "vector search angle".
   Both Claude steps go through the Message Batches API, so a run usually takes up to an hour longer than a direct call would.
4. **Publish**: the digest is committed to `digests/daily/YYYY/MM/YYYY-MM-DD.md` and posted to Discord. If nothing relevant came in, no digest is written and no notification is sent.

## Setup

1. Fork this repo
2. Add your `ANTHROPIC_API_KEY` as a repository secret (Settings → Secrets and variables → Actions)
3. Trigger a manual run from the Actions tab → **Daily Vector Search Digest** → Run workflow

## Discord notifications

To post each digest to a Discord Forum channel, create a channel webhook and add its URL as the `DISCORD_WEBHOOK_URL` repository secret. Each digest becomes one Forum post, rendered as native Discord Markdown with clickable paper titles and a GitHub link. If the secret is absent, digests are still generated and committed.

To test a notification without generating a new digest, run **Send Latest Digest to Discord** from the Actions tab. It sends the newest committed daily digest without changing the repository.

## Run locally

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/digest.py
```

## Tuning what gets flagged

- Edit `FOCUS_DESCRIPTION` in `scripts/consts.py`. It describes what counts as relevant and includes calibration examples.
- When the screen misses a paper you cared about, or flags one you didn't, add its arXiv id to `EXPECTED_RELEVANT` / `EXPECTED_IRRELEVANT` in `scripts/check_triage.py` and run:

  ```bash
  python scripts/check_triage.py
  ```

  It runs those papers through the same triage call and fails on any mismatch.

## Configuration

Constants in `scripts/consts.py`:

| Constant | Default | Description |
|---|---|---|
| `ARXIV_CATEGORIES` | 9 categories | arXiv categories to fetch (cross-listings included) |
| `LOOKBACK_DAYS` | 4 | arXiv submission window; overlap is deduped by `seen.json` |
| `HF_LOOKBACK_DAYS` | 2 | HuggingFace daily-papers window |
| `TRIAGE_MODEL` / `TRIAGE_EFFORT` | `claude-opus-5-5` / `low` | Model that screens every paper |
| `DIGEST_MODEL` / `DIGEST_EFFORT` | `claude-opus-5-5` / `high` | Model that writes the digest |
| `USE_BATCH_API` | `True` | Send Claude requests through the Message Batches API (50% cheaper, usually done within an hour). Refused or failed requests are retried directly at full price |
| `MIN_TRIAGE_SCORE` | 2 | Minimum 0–3 score to make the digest |
| `MAX_PAPERS_IN_DIGEST` | 25 | Cap on papers per digest, highest scores first |
| `SEEN_RETENTION_DAYS` | 30 | How long screened paper ids are remembered |
