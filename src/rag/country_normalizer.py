"""Country normalization to ISO 3166-1 alpha-2 codes.

Single source of truth for both index-time and query-time normalization.
Both sides write/match on `country_code` (e.g. "US"), eliminating the
"USA" vs "United States" vs "America" filter-mismatch class of bugs.

Uses `pycountry` for canonical ISO data (~250 countries) and a curated
demonym/abbreviation map for things pycountry doesn't cover ("american",
"british", "aussie", "uae", etc.).
"""

from __future__ import annotations

import difflib
import re
from functools import lru_cache

try:
    import pycountry
except ImportError as _e:
    pycountry = None
    _PYCOUNTRY_IMPORT_ERROR: ImportError | None = _e
else:
    _PYCOUNTRY_IMPORT_ERROR = None


# Demonyms, slang, and abbreviations not handled by pycountry.lookup().
# Keys are normalized via `_normalize_key` (lowercased, alpha-only).
DEMONYMS_AND_ABBREVIATIONS: dict[str, str] = {
    # United States
    "us": "US",
    "usa": "US",
    "america": "US",
    "american": "US",
    "americans": "US",
    "stateside": "US",
    "yank": "US",
    "yankee": "US",
    # United Kingdom
    "uk": "GB",
    "britain": "GB",
    "greatbritain": "GB",
    "british": "GB",
    "england": "GB",
    "english": "GB",
    "scotland": "GB",
    "scottish": "GB",
    "wales": "GB",
    "welsh": "GB",
    # United Arab Emirates
    "uae": "AE",
    # Other demonyms
    "australian": "AU",
    "australians": "AU",
    "aussie": "AU",
    "aussies": "AU",
    "canadian": "CA",
    "canadians": "CA",
    "chinese": "CN",
    "french": "FR",
    "german": "DE",
    "germans": "DE",
    "greek": "GR",
    "indian": "IN",
    "indians": "IN",
    "indonesian": "ID",
    "iranian": "IR",
    "iraqi": "IQ",
    "irish": "IE",
    "israeli": "IL",
    "italian": "IT",
    "italians": "IT",
    "japanese": "JP",
    "korean": "KR",
    "southkorean": "KR",
    "northkorean": "KP",
    "mexican": "MX",
    "mexicans": "MX",
    "moroccan": "MA",
    "dutch": "NL",
    "netherland": "NL",
    "pakistani": "PK",
    "peruvian": "PE",
    "philippine": "PH",
    "filipino": "PH",
    "polish": "PL",
    "portuguese": "PT",
    "romanian": "RO",
    "russian": "RU",
    "saudi": "SA",
    "spanish": "ES",
    "swedish": "SE",
    "swiss": "CH",
    "taiwanese": "TW",
    "thai": "TH",
    "turkish": "TR",
    "ukrainian": "UA",
    "vietnamese": "VN",
    "drc": "CD",
    "congolese": "CD",
}


def _normalize_key(s: str | None) -> str:
    return re.sub(r"[^a-z]", "", (s or "").strip().lower())


def _require_pycountry():
    if pycountry is None:
        raise RuntimeError(
            "pycountry is required for country normalization. "
            f"Install with `pip install pycountry`. ({_PYCOUNTRY_IMPORT_ERROR})"
        )


@lru_cache(maxsize=1)
def _iso2_to_name_map() -> dict[str, str]:
    _require_pycountry()
    return {c.alpha_2: c.name for c in pycountry.countries}


def to_iso2(raw: str | None, *, min_ratio: float = 0.86) -> str | None:
    """Map any country string (name, demonym, abbr, typo) to ISO 3166-1 alpha-2.

    Returns the alpha-2 code uppercased (e.g. "US"), or None if nothing matched
    confidently.
    """
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None

    iso2_to_name = _iso2_to_name_map()

    # Direct ISO2 match (case-insensitive).
    upper = s.upper()
    if len(upper) == 2 and upper in iso2_to_name:
        return upper

    key = _normalize_key(s)
    if not key:
        return None

    # Curated demonym / abbreviation map.
    if key in DEMONYMS_AND_ABBREVIATIONS:
        return DEMONYMS_AND_ABBREVIATIONS[key]

    # Plural strip: "Canadians" -> "canadian".
    if key.endswith("s") and len(key) > 3:
        singular = key[:-1]
        if singular in DEMONYMS_AND_ABBREVIATIONS:
            return DEMONYMS_AND_ABBREVIATIONS[singular]

    # pycountry canonical / official / common name lookup. Restricted to inputs
    # of >= 4 chars, OR uppercase 3-letter tokens (intentional ISO 3166-1 alpha-3),
    # so common short English words ("and", "the", "for") don't collide with
    # alpha-3 codes ("AND" = Andorra, "THE" — not a code, but the principle holds).
    if len(s) >= 4 or (len(s) == 3 and s.isupper()):
        try:
            match = pycountry.countries.lookup(s)
            return match.alpha_2
        except LookupError:
            pass

    # Cleaned-key match against canonical names.
    for code, name in iso2_to_name.items():
        if key == _normalize_key(name):
            return code

    # Fuzzy fallback against canonical names.
    best_code: str | None = None
    best_ratio = 0.0
    for code, name in iso2_to_name.items():
        ratio = difflib.SequenceMatcher(None, key, _normalize_key(name)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_code = code
    if best_code and best_ratio >= min_ratio:
        return best_code
    return None


def iso2_to_name(code: str | None) -> str | None:
    """Return the canonical English name for an ISO 3166-1 alpha-2 code, or None."""
    if not code:
        return None
    return _iso2_to_name_map().get(code.upper())


def extract_iso2_codes_from_text(text: str) -> list[str]:
    """Return ISO2 codes mentioned in free text (deterministic order, deduped)."""
    if not text:
        return []
    text_lower = text.lower()
    iso2_to_name = _iso2_to_name_map()
    found: list[str] = []
    seen: set[str] = set()

    # Match canonical names first; longest-first to avoid sub-word overlap
    # (e.g. "United States" before any shorter overlapping token).
    for code, name in sorted(iso2_to_name.items(), key=lambda kv: (-len(kv[1]), kv[1])):
        pattern = rf"\b{re.escape(name.lower())}\b"
        if re.search(pattern, text_lower) and code not in seen:
            seen.add(code)
            found.append(code)

    # Tokenize and try aliases / pycountry / fuzzy on each token.
    # Keep '-' as the last char in the class to avoid regex range ambiguity.
    tokens = re.findall(r"[A-Za-z][A-Za-z.'-]*[A-Za-z]?", text)
    for token in tokens:
        key = _normalize_key(token)
        if not key:
            continue
        # 2-letter lowercase tokens ("in", "or", "of", "us") are almost always
        # prepositions in free text. Only honor 2-letter tokens when they're
        # uppercase, signalling intentional ISO 3166-1 alpha-2 / abbreviation use.
        if len(key) == 2 and not token.isupper():
            continue
        # 3-letter all-lowercase tokens that are common English words also slip
        # through fuzzy matching; the to_iso2 alpha-3 path already filters these,
        # but we avoid the call when the token is too short to be informative.
        if len(key) < 2:
            continue
        code = to_iso2(token)
        if code and code not in seen:
            seen.add(code)
            found.append(code)

    return found
