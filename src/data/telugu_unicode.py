from typing import Tuple, Dict

VOWELS = {
    "a": "అ", "aa": "ఆ", "e": "ఇ", "ee": "ఈ", "i": "ఇ", "ii": "ఈ",
    "u": "ఉ", "oo": "ఊ", "uu": "ఊ", "ru": "ఋ", "roo": "ౠ",
    "e1": "ఎ", "e2": "ఏ", "ae": "ఏ", "ai": "ఐ", "o": "ఒ", "o1": "ఒ",
    "o2": "ఓ", "au": "ఔ", "am": "అం", "ah": "అః"
}

CONSONANTS = {
    "k": "క", "ka": "క", "kh": "ఖ", "kha": "ఖ", "g": "గ", "ga": "గ", "gh": "ఘ", "gha": "ఘ", "in": "ఙ",
    "c": "చ", "ch": "ఛ", "cha": "చ", "ch1": "ఛ", "j": "జ", "ja": "జ", "jh": "ఝ", "jha": "ఝ", "in1": "ఞ",
    "t": "ట", "ta": "ట", "th": "ఠ", "tha": "ఠ", "d": "డ", "da": "డ", "dh": "ఢ", "dha": "ఢ", "ana": "ణ", "an": "ణ",
    "t1": "త", "th1": "థ", "d1": "ద", "dh1": "ధ", "n": "న", "na": "న",
    "p": "ప", "P": "ప", "ph": "ఫ", "Ph": "ఫ", "b": "బ", "bh": "భ", "m": "మ",
    "y": "య", "r": "ర", "l": "ల", "v": "వ", "s": "శ", "sh": "ష", "s1": "స", "sa": "స", "h": "హ",
    "la": "ళ", "la1": "ళ", "ksha": "క్ష", "RR": "ఱ", "rra": "ఱ"
}

VOWEL_SIGNS = {
    "a": "", "aa": "ా", "RRA": "ా", "e": "ి", "ee": "ీ", "i": "ి", "ii": "ీ",
    "u": "ు", "oo": "ూ", "uu": "ూ", "ru": "ృ", "roo": "ౄ", "r": "ృ",
    "e1": "ె", "e2": "ే", "ai": "ై", "o1": "ొ", "o2": "ో", "au": "ౌ", "am": "ం", "ah": "ః"
}

def map_class_to_telugu(cls_name: str) -> Tuple[str, str, str]:
    clean_cls = cls_name.replace("/", "__")
    parts = clean_cls.split("__")
    category = parts[0].capitalize()

    if category.lower() == "achulu":
        v = parts[1] if len(parts) > 1 else ""
        glyph = VOWELS.get(v.lower(), "అ")
        return glyph, f"Vowel ({v})", "Achulu (Vowels)"

    elif category.lower() == "hallulu":
        c = parts[1] if len(parts) > 1 else ""
        glyph = CONSONANTS.get(c, CONSONANTS.get(c.lower(), "క"))
        return glyph, f"Consonant ({c})", "Hallulu (Consonants)"

    elif category.lower() == "othulu":
        c = parts[1] if len(parts) > 1 else ""
        base = CONSONANTS.get(c, CONSONANTS.get(c.lower(), "క"))
        glyph = f"్{base}"
        return glyph, f"Conjunct Mark ({c})", "Othulu (Conjuncts)"

    elif category.lower() == "guninthamulu":
        c = parts[1] if len(parts) > 1 else "ka"
        v = parts[2] if len(parts) > 2 else "a"
        base = CONSONANTS.get(c, CONSONANTS.get(c.lower(), "క"))
        sign = VOWEL_SIGNS.get(v.lower(), VOWEL_SIGNS.get(v, ""))
        glyph = f"{base}{sign}"
        return glyph, f"Modified Form ({c} + {v})", "Guninthamulu (Consonant+Vowel)"

    return cls_name, cls_name, "Other"
