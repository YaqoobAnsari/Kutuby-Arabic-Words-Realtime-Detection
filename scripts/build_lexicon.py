#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-time Quranic Lexicon Extractor (fully automatic)

Fetches the Uthmani (diacritized) Quran text from a public source, extracts
the unique word forms, and writes `data/quranic_lexicon.json` for the API to
load at startup. No manual download required.

Usage
-----
    python scripts/build_lexicon.py

Optional flags:
    --source URL    Override the default source URL
    --output PATH   Override the default output path

Public sources considered
-------------------------
- alquran.cloud public JSON API (used by default — no auth, single request,
  served from Cloudflare, sources its data from Tanzil)
  https://api.alquran.cloud/v1/quran/quran-uthmani
- tanzil.net direct text download (requires copyright acceptance flow)
- HuggingFace dataset `arbml/quran` or `tarteel-ai/everyayah` text fields
  (heavier dependency; not used here for text-only extraction)

Output
------
`data/quranic_lexicon.json`:

    {
      "version": "1.0",
      "source": "<source identifier>",
      "generated_at": "<ISO timestamp>",
      "count": <int>,
      "words": [
        {"diacritized": "اللَّهِ", "normalized": "الله"},
        ...
      ]
    }

`normalized` is the dedup key. The first-seen diacritized form is kept as the
canonical surface form — Tarteel's tokenizer can emit the diacritized form, so
this matches its output space.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout so Arabic text and arrows print correctly on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Reuse the production normalizer so the dedup key matches /verify_word semantics
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.arabic_utils import normalize_arabic_text


# `quran-simple-enhanced` uses regular alef (U+0627) with full diacritics — matches
# the orthography Tarteel's whisper-base-ar-quran emits. The default `quran-uthmani`
# edition uses super-alef (U+0670 `ٰ`) and would tokenize differently than the model's
# output, breaking lexicon-constrained beam decode.
DEFAULT_SOURCE_URL = "https://api.alquran.cloud/v1/quran/quran-simple-enhanced"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "quranic_lexicon.json"


def _fetch_quran_json(url: str) -> dict:
    print(f"build_lexicon: fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "kutuby-lexicon-builder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} from {url}")
        return json.loads(resp.read().decode("utf-8"))


def _extract_verses(payload: dict) -> list[str]:
    """Pull the verse text list out of the alquran.cloud response shape."""
    data = payload.get("data") or {}
    surahs = data.get("surahs") or []
    verses: list[str] = []
    for surah in surahs:
        for ayah in surah.get("ayahs", []):
            text = ayah.get("text")
            if text:
                verses.append(text)
    return verses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE_URL, help="Source URL (default: alquran.cloud Uthmani)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    args = parser.parse_args(argv)

    try:
        payload = _fetch_quran_json(args.source)
    except Exception as e:
        print(f"build_lexicon: ERROR — failed to fetch {args.source}: {e}", file=sys.stderr)
        return 2

    verses = _extract_verses(payload)
    if not verses:
        print("build_lexicon: ERROR — no verses extracted; response shape may have changed", file=sys.stderr)
        return 2
    print(f"build_lexicon: {len(verses)} verses fetched")

    # Extract unique word forms
    seen_normalized: dict[str, str] = {}  # normalized -> first-seen diacritized form
    raw_token_count = 0
    for line in verses:
        for token in line.split():
            raw_token_count += 1
            normalized = normalize_arabic_text(token)
            if not normalized:
                continue
            if normalized not in seen_normalized:
                seen_normalized[normalized] = token

    print(f"build_lexicon: {raw_token_count} raw tokens → {len(seen_normalized)} unique normalized words")

    words = [
        {"diacritized": diacritized, "normalized": normalized}
        for normalized, diacritized in sorted(seen_normalized.items())
    ]

    output_payload = {
        "version": "1.0",
        "source": args.source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(words),
        "words": words,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"build_lexicon: wrote {len(words)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
