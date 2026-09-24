# """IO, country-agnostic normalisation, macro-F0.5 metric."""
# import re, unicodedata
# import numpy as np, pandas as pd

# LEGAL = {"inc", "corp", "ltd", "pvt", "llc", "llp", "co", "plc", "sa", "sas", "sarl", "gmbh", "the", "and", "of"}
# ABBR = {"corporation": "corp", "incorporated": "inc", "limited": "ltd", "private": "pvt", "company": "co",
#         "road": "rd", "street": "st", "avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd",
#         "drive": "dr", "lane": "ln", "suite": "ste", "floor": "fl", "building": "bldg", "near": "nr",
#         "opposite": "opp", "highway": "hwy", "north": "n", "south": "s", "east": "e", "west": "w"}
# _SEP = re.compile(r"[\W_]+")
# _POSTAL = re.compile(r"\b\d{5,6}\b")  # US/FR 5-digit, IN 6-digit; no country hard-coding


# def read_tsv(p):
#     return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)


# def norm(s):
#     s = unicodedata.normalize("NFKD", s or "")
#     s = "".join(c for c in s if not unicodedata.combining(c)).lower().replace("&", " and ")
#     return " ".join(ABBR.get(t, t) for t in _SEP.sub(" ", s).split())


# def core(n):
#     t = [w for w in n.split() if w not in LEGAL]
#     return " ".join(t) or n


# def postal(a):
#     m = _POSTAL.findall(a)
#     return m[-1] if m else ""


# def prep(df):
#     df = df.copy()
#     df["n"] = df.business_name.map(norm)
#     df["nc"] = df.n.map(core)
#     df["a"] = df.business_address.map(norm)
#     df["pc"] = df.a.map(postal)
#     df["country"] = df.country.str.strip().str.lower()
#     return df.reset_index(drop=True)


# def load(root, split):
#     return [prep(read_tsv(f"{root}/{split}/{split}_source{i}.tsv")) for i in (1, 2, 3)]


# def read_gt(path):
#     g = read_tsv(path)
#     return {r.source1_entity_id: {x for x in r.matched_entity_ids.split(",") if x} for r in g.itertuples()}


# def f05_macro(truth, pred):
#     sc = []
#     for k, t in truth.items():
#         p = pred.get(k, set())
#         if not t and not p:
#             sc.append(1.0); continue
#         tp = len(t & p)
#         if tp == 0:
#             sc.append(0.0); continue
#         pr, rc = tp / len(p), tp / len(t)
#         sc.append(1.25 * pr * rc / (0.25 * pr + rc))
#     return float(np.mean(sc))


"""IO, script-aware normalisation, macro-F0.5 metric.
Changes vs v1: keeps Indic/other combining marks (old regex split Devanagari/Tamil words at vowel signs),
strips accents only on Latin letters, drops literal 'null' address tokens, unwraps URL-style names."""
import re, unicodedata
import numpy as np, pandas as pd

LEGAL = {"inc", "corp", "ltd", "pvt", "llc", "llp", "co", "plc", "sa", "sas", "sarl", "gmbh", "the", "and", "of"}
ABBR = {"corporation": "corp", "incorporated": "inc", "limited": "ltd", "private": "pvt", "company": "co",
        "road": "rd", "street": "st", "avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd",
        "drive": "dr", "lane": "ln", "suite": "ste", "floor": "fl", "building": "bldg", "near": "nr",
        "opposite": "opp", "highway": "hwy", "north": "n", "south": "s", "east": "e", "west": "w"}
_NULLS = {"null", "none", "nan", "na", "n/a"}
_SEP = re.compile(r"[^\w\u0300-\u036f\u0591-\u05c7\u0610-\u061a\u064b-\u065f\u0900-\u0dff\u0e31-\u0e4e\u200c\u200d]+")
_URL = re.compile(r"^(?:https?://)?(?:www\.)?(.+?)\.(?:co\.in|com\.au|com|net|org|in|co|io|fr|us|biz|info)(?:/.*)?$")
_POSTAL = re.compile(r"\b\d{5,6}\b")  # US/FR 5-digit, IN 6-digit; no country hard-coding


def read_tsv(p):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)


def script_of(s):
    sc = set()
    for ch in s:
        if ch.isalpha():
            o = ord(ch)
            sc.add("latin" if o < 0x250 or 0x1e00 <= o <= 0x1eff else "deva" if 0x900 <= o <= 0x97f
                   else "tamil" if 0xb80 <= o <= 0xbff else "indic_other" if 0x980 <= o <= 0xdff else "other")
    return "none" if not sc else next(iter(sc)) if len(sc) == 1 else "mixed"


def is_url(s):
    s = (s or "").strip().lower()
    return " " not in s and bool(_URL.match(s))


def _deacc(s):
    if s.isascii():
        return s
    out = []
    for c in unicodedata.normalize("NFC", s):
        if ord(c) < 0x370 or 0x1e00 <= ord(c) <= 0x1eff:  # Latin only
            out.extend(x for x in unicodedata.normalize("NFKD", c) if not unicodedata.combining(x))
        else:
            out.append(c)
    return "".join(out)


def norm(s, kind="name"):
    s = _deacc(s or "").lower().replace("&", " and ").replace("_", " ").replace("\u0964", " ").replace("\u0965", " ")
    if kind == "name":
        m = _URL.match(s.strip()) if " " not in s.strip() else None
        if m:
            s = m.group(1)
    toks = _SEP.sub(" ", s).split()
    if kind == "addr":
        toks = [t for t in toks if t not in _NULLS]
    return " ".join(ABBR.get(t, t) for t in toks)


def core(n):
    t = [w for w in n.split() if w not in LEGAL]
    return " ".join(t) or n


def postal(a):
    m = _POSTAL.findall(a)
    return m[-1] if m else ""


def prep(df):
    df = df.copy()
    df["n"] = df.business_name.map(norm)
    df["nc"] = df.n.map(core)
    df["a"] = df.business_address.map(lambda x: norm(x, "addr"))
    df["pc"] = df.a.map(postal)
    df["country"] = df.country.str.strip().str.lower()
    return df.reset_index(drop=True)


def load(root, split):
    return [prep(read_tsv(f"{root}/{split}/{split}_source{i}.tsv")) for i in (1, 2, 3)]


def read_gt(path):
    g = read_tsv(path)
    return {r.source1_entity_id: {x for x in r.matched_entity_ids.split(",") if x} for r in g.itertuples()}


def f05_macro(truth, pred):
    sc = []
    for k, t in truth.items():
        p = pred.get(k, set())
        if not t and not p:
            sc.append(1.0); continue
        tp = len(t & p)
        if tp == 0:
            sc.append(0.0); continue
        pr, rc = tp / len(p), tp / len(t)
        sc.append(1.25 * pr * rc / (0.25 * pr + rc))
    return float(np.mean(sc))