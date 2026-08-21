"""Telugu Unicode mapping module with exhaustive coverage for all 630 dataset class names."""

from typing import Tuple, Dict

# Mapping for 16 Telugu Vowels (Achulu)
VOWELS: Dict[str, str] = {
    "a": "అ", "aa": "ఆ", "i": "ఇ", "ii": "ఈ", "u": "ఉ", "uu": "ఊ",
    "ru": "ఋ", "ruu": "ౠ", "e": "ఎ", "ee": "ఏ", "ai": "ఐ", "o": "ఒ",
    "oo": "ఓ", "au": "ఔ", "ao": "ఔ", "am": "అం", "ah": "అః",
}

# Mapping for all Consonant abbreviations used across Hallulu, Guninthamulu, and Othulu
CONSONANTS: Dict[str, str] = {
    # Velar
    "k": "క", "ka": "క", "khh": "ఖ", "kh": "ఖ", "kha": "ఖ",
    "g": "గ", "ga": "గ", "gh": "ఘ", "gha": "ఘ", "gna": "ఙ",
    
    # Palatal
    "c": "చ", "ch": "ఛ", "cha": "చ", "chh": "ఛ",
    "j": "జ", "ja": "జ", "jh": "ఝ", "jha": "ఝ", "jna": "ఞ",
    
    # Retroflex
    "t": "ట", "ta": "ట", "tt": "ట్ట", "th": "ఠ", "tha": "ఠ", "thah": "ఠః",
    "d": "డ", "da": "డ", "dh": "ఢ", "dha": "ఢ", "ana": "ణ", "an": "్ణ", "nn": "ణ్ణ",
    
    # Dental
    "th_dental": "త", "tha_dental": "త", "d_dental": "ద", "da_dental": "ద",
    "n": "న", "na": "న",
    
    # Labial
    "p": "ప", "P": "ప", "pa": "ప", "ph": "ఫ", "Ph": "ఫ", "pha": "ఫ",
    "b": "బ", "ba": "బ", "bh": "భ", "bha": "భ", "m": "మ", "ma": "మ",
    
    # Semi-vowels & Liquids
    "y": "య", "ya": "య",
    "r": "ర", "ra": "ర", "rr": "ఱ", "RR": "ఱ",
    "l": "ల", "la": "ల", "ll": "ళ",
    "v": "వ", "va": "వ",
    
    # Sibilants & Aspirate
    "sh": "శ", "sha": "ష",
    "s": "స", "sa": "స",
    "h": "హ", "ha": "హ",
    "ks": "క్ష", "ksh": "క్ష", "z": "్క"
}

# Diacritic vowel modifiers for Guninthamulu
VOWEL_SIGNS: Dict[str, str] = {
    "a": "", "aa": "ా", "i": "ి", "ii": "ీ", "u": "ు", "uu": "ూ",
    "ru": "ృ", "ruu": "ౄ", "e": "ె", "ee": "ే", "ai": "ై",
    "o": "ొ", "oo": "ో", "au": "ౌ", "ow": "ౌ", "am": "ం", "ah": "ః",
    "m": "ం", "r": "ృ", "rrr": "ౄ", "R": "ృ", "RRA": "ా", "RRI": "ి", "RRII": "ీ", "RRU": "ు", "RRUU": "ూ",
    "rre": "ె", "rree": "ే", "rrai": "ై", "rro": "ొ", "rroo": "ో", "rrow": "ౌ", "rrm": "ం", "rrah": "ః", "rru": "ు", "rruu": "ూ"
}


def map_class_to_telugu(class_name: str) -> Tuple[str, str, str]:
    """Maps a 630-dataset class name (e.g. 'hallulu__ka', 'Guninthamulu__kha__ku')
    to (glyph, description, category).
    """
    parts = class_name.replace("/", "__").split("__")
    if not parts:
        return "❓", "Unknown", "Unknown"
        
    cat_raw = parts[0].lower()
    
    # 1. Achulu (Vowels)
    if cat_raw == "achulu":
        v_key = parts[1].lower() if len(parts) > 1 else "a"
        glyph = VOWELS.get(v_key, "అ")
        desc = f"Vowel ({v_key})"
        return glyph, desc, "Vowel (అచ్చులు)"
        
    # 2. Hallulu (Consonants)
    elif cat_raw == "hallulu":
        c_key = parts[1] if len(parts) > 1 else "ka"
        glyph = CONSONANTS.get(c_key, CONSONANTS.get(c_key.lower(), "క"))
        desc = f"Consonant ({c_key})"
        return glyph, desc, "Consonant (హల్లులు)"
        
    # 3. Guninthamulu (Vowel-Modified Forms)
    elif cat_raw == "guninthamulu":
        c_key = parts[1] if len(parts) > 1 else "ka"
        v_key = parts[2] if len(parts) > 2 else "a"
        
        # Dataset-specific anomaly: 'kha' folder is 'క' (ka), and 'khh' folder is 'ఖ' (kha)
        if c_key.lower() == "kha":
            base_glyph = "క"
            c_desc = "ka"
        elif c_key.lower() == "khh":
            base_glyph = "ఖ"
            c_desc = "kha"
        elif c_key.lower() == "ch":
            base_glyph = "ఛ"
            c_desc = "chha"
        elif c_key.lower() == "th":
            base_glyph = "ఠ"
            c_desc = "tha"
        elif c_key.lower() == "dh":
            base_glyph = "ఢ"
            c_desc = "dha"
        elif c_key.lower() == "sh":
            base_glyph = "శ"
            c_desc = "sha"
        elif c_key.lower() == "sha":
            base_glyph = "ష"
            c_desc = "ssha"
        elif c_key.lower() == "rr":
            base_glyph = "ఱ"
            c_desc = "rra"
        else:
            base_glyph = CONSONANTS.get(c_key, CONSONANTS.get(c_key.lower(), "క"))
            c_desc = c_key
            
        v_sign = VOWEL_SIGNS.get(v_key.lower(), VOWEL_SIGNS.get(v_key, ""))
        glyph = f"{base_glyph}{v_sign}"
        desc = f"Modified ({c_desc} + {v_key})"
        return glyph, desc, "Modified Form (గుణింతాలు)"
        
    # 4. Othulu (Conjunct Consonant Marks)
    elif cat_raw == "othulu":
        c_key = parts[1] if len(parts) > 1 else "k"
        base_glyph = CONSONANTS.get(c_key, CONSONANTS.get(c_key.lower(), "క"))
        glyph = f"్{base_glyph}"
        desc = f"Conjunct Mark ({c_key})"
        return glyph, desc, "Conjunct Mark (ఒత్తులు)"
        
    else:
        return class_name, class_name, "Other"
