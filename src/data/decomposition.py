"""Telugu Grapheme Decomposition Module.

Decomposes complex 630 Telugu aksharas into 3 independent classification targets:
1. Base Akshara (52 classes: 16 Vowels + 36 Consonants)
2. Vowel Modifier / Gunintham Sign (16 classes)
3. Subscript Conjunct / Othu / Vattu (37 classes: None + 36 Consonants)
"""

import json
from pathlib import Path
from typing import Dict, Tuple, List, Any


# 1. 52 Canonical Telugu Base Aksharas
BASE_LETTERS: List[str] = [
    # Achulu (Vowels) - 16
    "అ", "ఆ", "ఇ", "ఈ", "ఉ", "ఊ", "ఋ", "ౠ",
    "ఎ", "ఏ", "ఐ", "ఒ", "ఓ", "ఔ", "అం", "అః",
    # Hallulu (Consonants) - 36
    "క", "ఖ", "గ", "ఘ", "ఙ",
    "చ", "ఛ", "జ", "ఝ", "ఞ",
    "ట", "ఠ", "డ", "ఢ", "ణ",
    "త", "థ", "ద", "ధ", "న",
    "ప", "ఫ", "బ", "భ", "మ",
    "య", "ర", "ఱ", "ల", "ళ", "వ",
    "శ", "ష", "స", "హ", "క్ష"
]

# 2. 16 Canonical Vowel Modifiers (Guninthamulu)
VOWEL_MODIFIERS: List[str] = [
    "none",   # తలకట్టు / అ (క)
    "aa",     # దీర్ఘం ా (కా)
    "i",      # గుడి ి (కి)
    "ii",     # గుడిదీర్ఘం ీ (కీ)
    "u",      # కొమ్ము ు (కు)
    "uu",     # కొమ్ముదీర్ఘం ూ (కూ)
    "ru",     # వట్రుసుడి ృ (కృ)
    "ruu",    # వట్రుసుడి దీర్ఘం ౄ (కౄ)
    "e",      # ఎత్వం ె (కె)
    "ee",     # ఏత్వం ే (కే)
    "ai",     # ఐత్వం ై (కై)
    "o",      # ఒత్వం ొ (కొ)
    "oo",     # ఓత్వం ో (కో)
    "au",     # ఔత్వం ౌ (కౌ)
    "am",     # సున్నా ం (కం)
    "ah",     # విసర్గ ః (కః)
]

# 3. 37 Subscript Conjuncts (Othulu / Vattulu)
CONJUNCT_MODIFIERS: List[str] = [
    "none",   # Standard character (no subscript)
    "k", "kh", "g", "gh", "gna",
    "c", "ch", "j", "jh", "jna",
    "t", "th", "d", "dh", "ana",
    "th_dental", "d_dental", "dh_dental", "n",
    "p", "ph", "b", "bh", "m",
    "y", "r", "l", "v", "sh", "sha", "s", "h", "ll", "ks", "rr", "z"
]

BASE_MAP: Dict[str, int] = {letter: idx for idx, letter in enumerate(BASE_LETTERS)}
MOD_MAP: Dict[str, int] = {mod: idx for idx, mod in enumerate(VOWEL_MODIFIERS)}
VATTU_MAP: Dict[str, int] = {v: idx for idx, v in enumerate(CONJUNCT_MODIFIERS)}


# Alias dictionary for dataset folder quirks
CONSONANT_ALIASES = {
    "kha": "క", "khh": "ఖ", "kh": "ఖ", "ka": "క", "k": "క",
    "ga": "గ", "g": "గ", "gha": "ఘ", "gh": "ఘ",
    "cha": "చ", "ch": "ఛ", "chh": "ఛ", "c": "చ",
    "ja": "జ", "j": "జ", "jha": "ఝ", "jh": "ఝ",
    "ta": "ట", "t": "ట", "tt": "ట్ట", "tha": "ఠ", "th": "ఠ", "thah": "ఠః",
    "da": "డ", "d": "డ", "dha": "ఢ", "dh": "ఢ", "ana": "ణ", "an": "్ణ",
    "th_dental": "త", "tha_dental": "త", "d_dental": "ద", "da_dental": "ద",
    "na": "న", "n": "న",
    "pa": "ప", "p": "ప", "P": "ప", "pha": "ఫ", "ph": "ఫ", "Ph": "ఫ",
    "ba": "బ", "b": "బ", "bha": "భ", "bh": "భ", "ma": "మ", "m": "మ",
    "ya": "య", "y": "య",
    "ra": "ర", "r": "ర", "rr": "ఱ", "RR": "ఱ",
    "la": "ల", "l": "ల", "ll": "ళ",
    "va": "వ", "v": "వ",
    "sha": "ష", "sh": "శ",
    "sa": "స", "s": "స",
    "ha": "హ", "h": "హ",
    "ksh": "క్ష", "ks": "క్ష", "z": "క"
}

VOWEL_ALIASES = {
    "a": "అ", "aa": "ఆ", "i": "ఇ", "ii": "ఈ", "u": "ఉ", "uu": "ఊ",
    "ru": "ఋ", "ruu": "ౠ", "e": "ఎ", "ee": "ఏ", "ai": "ఐ",
    "o": "ఒ", "oo": "ఓ", "au": "ఔ", "ao": "ఔ", "am": "అం", "ah": "అః"
}

MOD_ALIASES = {
    "a": "none", "aa": "aa", "i": "i", "ii": "ii", "u": "u", "uu": "uu",
    "ru": "ru", "ruu": "ruu", "e": "e", "ee": "ee", "ai": "ai",
    "o": "o", "oo": "oo", "au": "au", "ow": "au", "am": "am", "ah": "ah",
    "m": "am", "r": "ru", "rrr": "ruu", "R": "ru", "RRA": "aa", "RRI": "i", "RRII": "ii",
    "RRU": "u", "RRUU": "uu", "rre": "e", "rree": "ee", "rrai": "ai", "rro": "o", "rroo": "oo",
    "rrow": "au", "rrm": "am", "rrah": "ah", "rru": "u", "rruu": "uu"
}

VATTU_ALIASES = {
    "an": "ana", "ana": "ana", "nn": "ana",
    "b": "b", "ba": "b", "bh": "bh", "bha": "bh",
    "c": "c", "ca": "c", "ch": "ch", "cha": "ch", "chh": "ch",
    "d": "d", "da": "d_dental", "dh": "dh", "dha": "dh_dental",
    "g": "g", "ga": "g", "gh": "gh", "gha": "gh",
    "h": "h", "ha": "h",
    "j": "j", "ja": "j", "jh": "jh", "jha": "jh",
    "k": "k", "ka": "k", "kh": "kh", "kha": "kh", "ksh": "ks", "ks": "ks",
    "l": "l", "la": "l", "ll": "ll", "lla": "ll",
    "m": "m", "ma": "m",
    "n": "n", "na": "n",
    "p": "p", "pa": "p", "ph": "ph", "pha": "ph",
    "r": "r", "ra": "r", "rr": "rr", "rra": "rr",
    "s": "s", "sa": "s", "sh": "sh", "sha": "sha",
    "t": "t", "ta": "t", "tt": "t", "th": "th", "tha": "th_dental", "th_dental": "th_dental",
    "v": "v", "va": "v",
    "y": "y", "ya": "y",
    "z": "z"
}


def decompose_class_name(class_name: str) -> Tuple[int, int, int]:
    """Decomposes any canonical 630 class name into (base_idx, modifier_idx, vattu_idx).
    
    Args:
        class_name: String like 'Guninthamulu__kha__ki', 'hallulu__ka', 'achulu__a', 'othulu__v'
        
    Returns:
        (base_idx, modifier_idx, vattu_idx) as integer indices.
    """
    parts = class_name.replace("/", "__").split("__")
    cat = parts[0].lower()
    
    # Defaults
    base_letter = "క"
    modifier = "none"
    vattu = "none"
    
    if cat == "achulu":
        v_key = parts[1].lower() if len(parts) > 1 else "a"
        base_letter = VOWEL_ALIASES.get(v_key, "అ")
        modifier = "none"
        vattu = "none"
        
    elif cat == "hallulu":
        c_key = parts[1] if len(parts) > 1 else "ka"
        base_letter = CONSONANT_ALIASES.get(c_key, CONSONANT_ALIASES.get(c_key.lower(), "క"))
        modifier = "none"
        vattu = "none"
        
    elif cat == "guninthamulu":
        c_key = parts[1] if len(parts) > 1 else "ka"
        v_key = parts[2] if len(parts) > 2 else "a"
        base_letter = CONSONANT_ALIASES.get(c_key, CONSONANT_ALIASES.get(c_key.lower(), "క"))
        modifier = MOD_ALIASES.get(v_key, "none")
        vattu = "none"
        
    elif cat == "othulu":
        c_key = parts[1] if len(parts) > 1 else "k"
        base_letter = CONSONANT_ALIASES.get(c_key, CONSONANT_ALIASES.get(c_key.lower(), "క"))
        modifier = "none"
        v_raw = c_key.lower()
        vattu = VATTU_ALIASES.get(c_key, VATTU_ALIASES.get(v_raw, "k"))
        
    base_idx = BASE_MAP.get(base_letter, 0)
    mod_idx = MOD_MAP.get(modifier, 0)
    vattu_idx = VATTU_MAP.get(vattu, 0)
    
    return base_idx, mod_idx, vattu_idx


def export_grapheme_maps(output_path: str = "outputs/grapheme_maps.json") -> Dict[str, Any]:
    """Exports all grapheme maps to JSON."""
    data = {
        "num_base_classes": len(BASE_LETTERS),
        "num_modifier_classes": len(VOWEL_MODIFIERS),
        "num_vattu_classes": len(CONJUNCT_MODIFIERS),
        "base_letters": BASE_LETTERS,
        "base_map": BASE_MAP,
        "vowel_modifiers": VOWEL_MODIFIERS,
        "mod_map": MOD_MAP,
        "conjunct_modifiers": CONJUNCT_MODIFIERS,
        "vattu_map": VATTU_MAP,
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
