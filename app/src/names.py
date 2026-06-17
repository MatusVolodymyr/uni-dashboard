"""Teacher-name canonicalization.

Student-typed names vary a lot: case, spacing, apostrophes, Latin homoglyphs,
initials vs full names, missing patronymics, typos, and several co-teachers in
one cell. This module:

  1. splits a cell into individual teacher pieces (co-teachers),
  2. normalizes each piece,
  3. builds a CONSERVATIVE identity key = (surname, first-initial, patronymic-initial),
  4. clusters pieces and picks the fullest spelling as the canonical display name.

Conservative rule: names that differ in patronymic initial are kept apart
(Супрун І.О. ≠ Супрун А.Г.). A piece missing its patronymic is only merged into
a fuller name when that (surname, first-initial) maps to exactly one patronymic.
"""
import re
from collections import Counter, defaultdict
from typing import Iterable

# Latin → Cyrillic homoglyphs (lowercase); applied only for KEYING, not display.
_HOMO = str.maketrans({
    "a": "а", "c": "с", "e": "е", "i": "і", "o": "о", "p": "р", "x": "х",
    "y": "у", "k": "к", "m": "м", "t": "т", "b": "ь", "h": "н", "n": "п",
})

_APOSTROPHES = "`'ʼʼ’‘ʻ"
_SEP_RE = re.compile(r"\s*(?:[,;/&+]|\bта\b|\bі\b|\bта,|\n)\s*", re.IGNORECASE)
_PAREN_RE = re.compile(r"\([^)]*\)")
_PATR_RE = re.compile(r"(вич|вна|ївна|івна|йович|евич|ович|инич|ішна|ишна)$", re.IGNORECASE)
_NOISE_WORDS = {"лектор", "практик", "викладач", "та", "і", "робота", "чудова", "дуже", "найкращий"}


def _strip_apostrophes(s: str) -> str:
    for a in _APOSTROPHES:
        s = s.replace(a, "'")
    return s


def _split_glued(piece: str) -> list[str]:
    """Split a space-glued multi-person piece (no explicit separator).

    Starts a new person when a capitalized full word (likely a new surname, not a
    patronymic) appears and the current chunk already holds ≥2 tokens.
    """
    toks = piece.split()
    chunks, cur = [], []
    for tok in toks:
        letters = re.sub(r"[^а-яіїєґA-Za-z]", "", tok)
        is_initial = "." in tok or len(letters) <= 1
        is_surname_like = (
            len(letters) >= 2 and tok[:1].isupper()
            and not is_initial and not _PATR_RE.search(tok.lower())
        )
        if cur and is_surname_like and len(cur) >= 2:
            chunks.append(" ".join(cur))
            cur = [tok]
        else:
            cur.append(tok)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def split_cell(raw: str) -> list[str]:
    """Split one raw cell into individual teacher-name pieces."""
    if not raw:
        return []
    s = str(raw).replace("\xa0", " ")
    s = _PAREN_RE.sub(" ", s)
    s = _strip_apostrophes(s)
    out = []
    for part in _SEP_RE.split(s):
        part = re.sub(r"\s+", " ", part).strip(" .,:;-")
        if not part:
            continue
        for p in _split_glued(part):
            p = p.strip(" .,:;-")
            if len(re.sub(r"[^а-яіїєґA-Za-z]", "", p)) >= 2:
                out.append(p)
    return out


def _norm_for_key(piece: str) -> str:
    s = _strip_apostrophes(piece).lower().replace("\xa0", " ")
    s = s.replace(".", " ")
    s = s.translate(_HOMO)
    s = re.sub(r"[^а-яіїєґ' -]", " ", s)   # keep cyrillic letters, apostrophe, hyphen
    s = re.sub(r"\s+", " ", s).strip()
    return s


def name_key(piece: str):
    """Return (surname, first_initial, patronymic_initial) or None if unusable."""
    norm = _norm_for_key(piece)
    toks = [t for t in norm.split() if t and t not in _NOISE_WORDS]
    if not toks:
        return None
    surname = toks[0]
    if len(surname) < 2:
        return None
    fi = toks[1][0] if len(toks) > 1 else ""
    pi = toks[2][0] if len(toks) > 2 else ""
    return (surname, fi, pi)


def _completeness(piece: str) -> tuple:
    """Higher = fuller, cleaner spelling.

    Rewards a full first name and a real patronymic; penalizes extra tokens
    (a sign of a glued co-teacher). Frequency breaks ties at the call site.
    """
    norm = _norm_for_key(piece)
    toks = norm.split()
    full_first = len(toks) > 1 and len(toks[1]) > 1
    full_patr = len(toks) > 2 and len(toks[2]) > 1 and bool(_PATR_RE.search(toks[2]))
    extra = max(0, len(toks) - 3)
    return (full_first + full_patr, -extra)


def _clean_display(piece: str) -> str:
    s = _strip_apostrophes(piece).replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip(" .,:;-")
    # Title-case each token, keep initials like "І." tidy
    out = []
    for tok in s.split():
        if len(tok) == 1:
            out.append(tok.upper() + ".")
        else:
            out.append(tok[0].upper() + tok[1:])
    return " ".join(out)


def build_canonical_map(cells: Iterable[str]) -> dict:
    """From all raw cells, build {raw_piece: canonical_display_name}.

    Two-pass conservative clustering keyed on (surname, first-initial, patr-initial).
    """
    # gather pieces with frequencies
    piece_counts = Counter()
    for cell in cells:
        for piece in split_cell(cell):
            piece_counts[piece] += 1

    # group keyed pieces; track which patronymics exist per (surname, first-initial)
    full_keys = defaultdict(Counter)        # (surname,fi,pi) -> Counter(piece->freq), pi != ''
    surname_fi_to_pis = defaultdict(set)    # (surname,fi) -> {pi!=''}
    stubs = {}                              # piece -> key  (pi == '')
    piece_to_key = {}

    for piece, freq in piece_counts.items():
        k = name_key(piece)
        if k is None:
            continue
        surname, fi, pi = k
        piece_to_key[piece] = k
        if pi:
            full_keys[k][piece] += freq
            surname_fi_to_pis[(surname, fi)].add(pi)
        else:
            stubs[piece] = k

    # attach stubs conservatively
    for piece, (surname, fi, pi) in stubs.items():
        if fi:
            pis = surname_fi_to_pis.get((surname, fi))
            if pis and len(pis) == 1:
                only_pi = next(iter(pis))
                full_keys[(surname, fi, only_pi)][piece] += piece_counts[piece]
                piece_to_key[piece] = (surname, fi, only_pi)
                continue
        else:
            # surname-only: attach only if the surname maps to exactly one identity
            surname_keys = [k2 for k2 in full_keys if k2[0] == surname]
            if len(surname_keys) == 1:
                tgt = surname_keys[0]
                full_keys[tgt][piece] += piece_counts[piece]
                piece_to_key[piece] = tgt
                continue
        # otherwise keep as its own group (ambiguous → not merged)
        full_keys[(surname, fi, pi)][piece] += piece_counts[piece]

    # choose canonical display per key
    key_display = {}
    for k, pieces in full_keys.items():
        best = max(pieces, key=lambda p: (_completeness(p), pieces[p]))
        key_display[k] = _clean_display(best)

    # build piece -> display map
    canonical = {}
    for piece in piece_counts:
        k = piece_to_key.get(piece)
        if k is None:
            continue
        canonical[piece] = key_display.get(k, _clean_display(piece))
    return canonical


def canonical_names(cell: str, canonical_map: dict) -> list[str]:
    """Resolve a raw cell to a deduplicated list of canonical teacher names."""
    seen, out = set(), []
    for piece in split_cell(cell):
        name = canonical_map.get(piece) or _clean_display(piece)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out
