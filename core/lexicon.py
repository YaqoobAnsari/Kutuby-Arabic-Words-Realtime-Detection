# -*- coding: utf-8 -*-
"""
Quranic Lexicon Trie + Whisper Constrained-Decoding Hook

Loads `data/quranic_lexicon.json` (produced by `scripts/build_lexicon.py`),
tokenizes every entry with the Whisper tokenizer, and builds a token-level
trie. The trie is queried during beam search via `prefix_allowed_tokens_fn`
so the decoder can only emit token sequences that spell a valid Quranic word.

Whisper specifics
-----------------
Whisper's decoder prepends a prompt (`<|startoftranscript|>`, language token,
task token, `<|notimestamps|>`) before any transcription tokens. The trie
operates on the **transcription tokens only**, so the `prefix_allowed_tokens_fn`
slices off the prompt before consulting the trie.

The same Arabic word can tokenize to slightly different token sequences
depending on leading whitespace (BPE merges differently). To stay robust,
each lexicon entry is inserted in multiple tokenization variants (with and
without a leading space).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TokenTrie:
    """A prefix trie over Whisper token IDs.

    Each path from root to a `is_terminal` node spells out one lexicon word.
    """

    __slots__ = ("children", "is_terminal")

    def __init__(self) -> None:
        self.children: dict[int, "TokenTrie"] = {}
        self.is_terminal: bool = False

    def insert(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        node = self
        for tok in token_ids:
            child = node.children.get(tok)
            if child is None:
                child = TokenTrie()
                node.children[tok] = child
            node = child
        node.is_terminal = True

    def find(self, prefix: list[int]) -> Optional["TokenTrie"]:
        """Return the node reached by walking `prefix`, or None if off-trie."""
        node = self
        for tok in prefix:
            child = node.children.get(tok)
            if child is None:
                return None
            node = child
        return node

    def allowed_next(self, prefix: list[int], eos_token_id: int) -> set[int]:
        """Token IDs valid as the next emission given `prefix` already emitted."""
        node = self.find(prefix)
        if node is None:
            return set()
        allowed = set(node.children.keys())
        if node.is_terminal:
            allowed.add(eos_token_id)
        return allowed

    def size(self) -> int:
        """Count the total number of terminal nodes (= unique lexicon entries)."""
        count = 1 if self.is_terminal else 0
        for child in self.children.values():
            count += child.size()
        return count


def _tokenize_word_variants(tokenizer, word: str) -> list[list[int]]:
    """Tokenize a word in multiple plausible forms so we match whatever the model emits.

    Whisper's BPE handles leading whitespace differently across positions; covering
    both variants avoids "model emits a token that's never in the trie" failures.
    """
    variants: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for surface in (word, " " + word):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if ids:
            key = tuple(ids)
            if key not in seen:
                seen.add(key)
                variants.append(ids)
    return variants


def build_trie_from_lexicon(tokenizer, lexicon_path: Path | str) -> TokenTrie:
    """Load the lexicon JSON and tokenize every entry into the trie.

    `tokenizer` should be the Whisper (or other) tokenizer that the model will
    decode with — token IDs must match the model's output space.
    """
    lexicon_path = Path(lexicon_path)
    payload = json.loads(lexicon_path.read_text(encoding="utf-8"))
    words = payload.get("words", [])
    if not words:
        raise ValueError(f"lexicon at {lexicon_path} has no 'words' entries")

    trie = TokenTrie()
    total_variants = 0
    for entry in words:
        word = entry.get("diacritized") or entry.get("normalized")
        if not word:
            continue
        for ids in _tokenize_word_variants(tokenizer, word):
            trie.insert(ids)
            total_variants += 1

    logger.info(
        "lexicon: loaded %d entries, %d tokenization variants into trie (%d terminal nodes)",
        len(words),
        total_variants,
        trie.size(),
    )
    return trie


def make_prefix_allowed_tokens_fn(
    trie: TokenTrie,
    prompt_token_count: int,
    eos_token_id: int,
    vocab_size: int,
) -> Callable[[int, "object"], list[int]]:
    """Build a `prefix_allowed_tokens_fn` for HuggingFace `model.generate()`.

    Behavior depends on the length of `input_ids` passed in:

    1. While the decoder is still emitting the special prompt prefix
       (e.g. for Whisper: <|sot|>, <|lang|>, <|task|>, <|notimestamps|>),
       we return the FULL vocabulary. The model's own LogitsProcessors
       (ForceTokensLogitsProcessor, SuppressTokensLogitsProcessor, etc.)
       handle picking the right forced token — we must not constrain them.

    2. Once we've passed the prompt (i.e. we're generating real transcription
       tokens), we consult the trie and return only the lexicon-valid tokens.

    `prompt_token_count` is the number of decoder prompt tokens (Whisper: 4).
    `vocab_size` is the model's tokenizer vocabulary size, needed so we can
    return `list(range(vocab_size))` to mean "no constraint, let other
    processors decide". Passing this list is mildly inefficient (~50k ints per
    forced-position call, 3 calls per beam) but only happens for the prompt
    prefix — generation is dominated by the constrained transcription steps.
    """
    full_vocab = list(range(vocab_size))

    def _fn(batch_id: int, input_ids) -> list[int]:
        try:
            seq = input_ids.tolist()
        except AttributeError:
            seq = list(input_ids)
        if len(seq) < prompt_token_count:
            # Still inside Whisper's prompt prefix — don't constrain.
            return full_vocab
        lexicon_prefix = seq[prompt_token_count:]
        allowed = trie.allowed_next(lexicon_prefix, eos_token_id)
        if not allowed:
            # Off-trie: nothing valid. Return EOS only to terminate cleanly.
            return [eos_token_id]
        return list(allowed)

    return _fn
