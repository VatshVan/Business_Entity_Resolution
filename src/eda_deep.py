"""Deep EDA. python -m src.eda_deep --data dataset_2pct --out eda_out [--test-data dataset]
Sections: 1 record profiles  2 match structure  3 pair noise taxonomy (name x address)  4 recoverability
          5 data-mined substitutions (abbreviations / translations)  6 name collisions (precision risk)  7 test glance"""
import argparse, os, re
from collections import Counter
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from src.common import load, read_gt, read_tsv, script_of, is_url

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
RN, NC, N, A, RA, C = range(6)
_LAND = re.compile(r"\b(?:near|opp|opposite|behind|beside|next to|adjacent)\b", re.I)
_POST = re.compile(r"\b\d{5,6}\b")
NAME_OK = {"exact", "suffix_only", "reorder", "token_subset", "scramble", "typo_small", "typo_heavy", "url_match"}
ADDR_OK = {"exact", "minor", "subset_reorder", "partial"}


def rec_dicts(df):
    return {e: (rn, nc, n, a, ra, c) for e, rn, nc, n, a, ra, c in
            zip(df.entity_id, df.business_name, df.nc, df.n, df.a, df.business_address, df.country)}


def name_class(q, t):
    if not t[N]:
        return "missing"
    if is_url(t[RN]) and not is_url(q[RN]):
        return "url_match" if fuzz.ratio(q[NC].replace(" ", ""), t[NC].replace(" ", "")) >= 85 else "url_diverged"
    sq, st = script_of(q[RN]), script_of(t[RN])
    if st == "mixed" and sq != "mixed":
        return "mixed_script"
    if sq != st:
        return "cross_script"
    c1, c2 = q[NC], t[NC]
    if q[N] == t[N]: return "exact"
    if c1 == c2: return "suffix_only"
    if sorted(c1.split()) == sorted(c2.split()): return "reorder"
    a, b = set(c1.split()), set(c2.split())
    if a <= b or b <= a: return "token_subset"
    if sorted(c1.replace(" ", "")) == sorted(c2.replace(" ", "")): return "scramble"
    r = fuzz.ratio(c1, c2)
    return "typo_small" if r >= 85 else "typo_heavy" if r >= 60 else "diverged"


def addr_class(q, t):
    a1, a2 = q[A], t[A]
    if not a2: return "missing"
    if script_of(t[RA]) not in ("latin", "none", "mixed"): return "native_script"
    if a1 == a2: return "exact"
    if fuzz.ratio(a1, a2) >= 90: return "minor"
    ts = fuzz.token_set_ratio(a1, a2)
    if ts >= 90: return "subset_reorder"
    if ts >= 70 or fuzz.token_sort_ratio(a1, a2) >= 70: return "partial"
    return "diverged"


def classify(q, t):
    return name_class(q, t), addr_class(q, t)


def rec_stats(df, nmax):
    d = df.sample(nmax, random_state=0) if len(df) > nmax else df
    nm, ad = d.business_name, d.business_address
    sc, sa = nm.map(script_of), ad.map(script_of)
    return {"n": len(df), "empty_name": (nm.str.strip() == "").mean(), "empty_addr": (ad.str.strip() == "").mean(),
            "addr_null": ad.str.contains(r"\bnull\b", case=False).mean(),
            "name_nonlatin": sc.isin(["deva", "tamil", "indic_other", "other"]).mean(), "name_mixed": (sc == "mixed").mean(),
            "name_url": nm.map(is_url).mean(), "name_upper": nm.str.isupper().mean(),
            "name_accent": ((~nm.map(str.isascii)) & (sc == "latin")).mean(), "name_ntok": nm.str.split().str.len().mean(),
            "addr_upper": ad.str.isupper().mean(), "addr_nonlatin": sa.isin(["deva", "tamil", "indic_other", "other"]).mean(),
            "addr_mixed": (sa == "mixed").mean(), "addr_postal": ad.str.contains(_POST).mean(),
            "addr_landmark": ad.str.contains(_LAND).mean(), "addr_digit": ad.str.contains(r"\d").mean(),
            "addr_ncomma": ad.str.count(",").mean()}


def pair_table(qd, td, gt, max_s1, rng):
    ids = [k for k, v in gt.items() if v and k in qd]
    if len(ids) > max_s1:
        ids = list(rng.choice(ids, max_s1, replace=False))
    rows, subs = [], {}
    for s in ids:
        q = qd[s]
        for t in gt[s]:
            if t not in td: continue
            r = td[t]
            nc, ac = classify(q, r)
            rows.append((s, t, t[:2], q[C], nc, ac, fuzz.ratio(q[NC], r[NC]), fuzz.token_set_ratio(q[A], r[A])))
            for kind, x, y in (("name", q[NC], r[NC]), ("addr", q[A], r[A])):
                sa, sb = set(x.split()) - set(y.split()), set(y.split()) - set(x.split())
                if len(sa) == 1 and len(sb) == 1:
                    subs.setdefault((q[C], kind), Counter())[(next(iter(sa)), next(iter(sb)))] += 1
    df = pd.DataFrame(rows, columns=["s1_id", "t_id", "src", "country", "name_cls", "addr_cls", "nc_ratio", "a_tset"])
    return df, subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_2pct"); ap.add_argument("--out", default="eda_out")
    ap.add_argument("--pairs", type=int, default=30000, help="max S1 entities used for pair analysis")
    ap.add_argument("--rec-sample", type=int, default=200000); ap.add_argument("--test-data", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); rng = np.random.RandomState(0)
    s1, s2, s3 = load(a.data, "train"); gt = read_gt(f"{a.data}/train/train_ground_truth.tsv")

    print("\n=== 1. RECORD PROFILES (share of records unless n / mean) ===")
    prof = pd.DataFrame({(n, c): rec_stats(d[d.country == c], a.rec_sample) for n, d in (("S1", s1), ("S2", s2), ("S3", s3))
                         for c in sorted(d.country.unique())}).T.round(3)
    print(prof); prof.to_csv(f"{a.out}/record_profiles.csv")

    print("\n=== 2. MATCH STRUCTURE ===")
    cty = s1.set_index("entity_id").country
    st = pd.DataFrame({"country": cty.reindex(list(gt)).values, "n2": [sum(x.startswith("S2-") for x in v) for v in gt.values()],
                       "n3": [sum(x.startswith("S3-") for x in v) for v in gt.values()]})
    st["only2"], st["only3"], st["both"], st["none"] = ((st.n2 > 0) & (st.n3 == 0)), ((st.n3 > 0) & (st.n2 == 0)), ((st.n2 > 0) & (st.n3 > 0)), ((st.n2 + st.n3) == 0)
    print(st.groupby("country").agg(n=("n2", "size"), mean_S2=("n2", "mean"), mean_S3=("n3", "mean"), only_S2=("only2", "mean"),
                                    only_S3=("only3", "mean"), both=("both", "mean"), singleton=("none", "mean")).round(3))
    print("matches per S1 histogram:", (st.n2 + st.n3).value_counts().sort_index().to_dict())
    owned = {x for v in gt.values() for x in v}
    for n, d in (("S2", s2), ("S3", s3)):
        print(f"{n} un-owned (distractor) share by country:", (~d.entity_id.isin(owned)).groupby(d.country).mean().round(3).to_dict())

    print("\n=== 3. PAIR NOISE TAXONOMY (matched pairs) ===")
    qd, td = rec_dicts(s1), rec_dicts(pd.concat([s2, s3]))
    pt, subs = pair_table(qd, td, gt, a.pairs, rng)
    pt.to_csv(f"{a.out}/pair_classes_sample.csv", index=False)
    for col in ("name_cls", "addr_cls"):
        print(f"\n{col} by source x country (% of pairs)")
        print((pd.crosstab([pt.country, pt.src], pt[col], normalize="index") * 100).round(1))
    print("\nexamples per name class (S1 | T):")
    for k, g in pt.groupby("name_cls"):
        for _, r in g.head(3).iterrows():
            print(f"  [{k}] {qd[r.s1_id][RN]!r} | {td[r.t_id][RN]!r}")
    print("\nexamples per address class:")
    for k, g in pt.groupby("addr_cls"):
        for _, r in g.head(2).iterrows():
            print(f"  [{k}] {qd[r.s1_id][RA]!r} | {td[r.t_id][RA]!r}")

    print("\n=== 4. RECOVERABILITY: what signal does each matched pair still carry? (% of pairs) ===")
    pt["name_state"] = pt.name_cls.map(lambda c: "name_ok" if c in NAME_OK else c)
    pt["addr_state"] = pt.addr_cls.map(lambda c: "addr_ok" if c in ADDR_OK else c)
    print((pd.crosstab(pt.name_state, pt.addr_state, normalize="all") * 100).round(2))
    both_bad = ((pt.name_state != "name_ok") & (pt.addr_state != "addr_ok"))
    print("pairs with NEITHER usable name NOR usable address:", round(both_bad.mean() * 100, 2), "%")
    print("  ...by name/addr state:\n", pt[both_bad].groupby(["name_state", "addr_state"]).size().sort_values(ascending=False).head(10))
    print("address-only pairs (name unusable, address ok):", round(((pt.name_state != "name_ok") & (pt.addr_state == "addr_ok")).mean() * 100, 2), "%")
    print("name-only pairs (address unusable, name ok):", round(((pt.name_state == "name_ok") & (pt.addr_state != "addr_ok")).mean() * 100, 2), "%")
    print("\nname-sim (ratio on core) percentiles of matched pairs:", np.percentile(pt.nc_ratio, [5, 10, 25, 50, 75]).round(1))

    print("\n=== 5. DATA-MINED SUBSTITUTIONS (1-token swaps in matched pairs; frequent ones = abbreviations/translations) ===")
    for (c, kind), cnt in subs.items():
        top = cnt.most_common(300)
        pd.DataFrame([(x, y, n) for (x, y), n in top], columns=["s1_token", "t_token", "count"]).to_csv(f"{a.out}/subs_{kind}_{c}.csv", index=False)
        print(f"[{c}/{kind}]", [(x, y, n) for (x, y), n in top[:20]])

    print("\n=== 6. NAME COLLISIONS (precision risk) ===")
    s1["nc_cnt"] = s1.groupby(["country", "nc"]).entity_id.transform("size")
    print("share of S1 whose core name is shared with another S1:", (s1.nc_cnt > 1).groupby(s1.country).mean().round(3).to_dict())
    pool = pd.concat([s2, s3]); idx = pool.groupby(["country", "nc"]).entity_id.apply(list).to_dict()
    samp = [k for k in gt if k in qd]; samp = list(rng.choice(samp, min(20000, len(samp)), replace=False))
    foreign = [len([t for t in idx.get((qd[s][C], qd[s][NC]), []) if t not in gt[s]]) for s in samp]
    print("S1 with >=1 NON-matching S2/S3 record of identical core name:", round(np.mean(np.array(foreign) > 0), 3),
          "| mean count:", round(float(np.mean(foreign)), 3), "(at this subset's density)")
    for c in sorted(s1.country.unique()):
        tk = Counter(w for x in s1[s1.country == c].nc for w in x.split())
        print(f"top name tokens [{c}]:", tk.most_common(25))
    print("most repeated S1 core names:", s1.groupby(["country", "nc"]).size().sort_values(ascending=False).head(10).to_dict())

    if a.test_data and os.path.isdir(f"{a.test_data}/test"):
        print("\n=== 7. TEST GLANCE (unlabelled; note the unseen country) ===")
        ts = {i: read_tsv(f"{a.test_data}/test/test_source{i}.tsv") for i in (1, 2, 3)}
        print({f"S{i}": d.country.value_counts().to_dict() for i, d in ts.items()})
        tp = pd.DataFrame({(f"S{i}", c): rec_stats(d[d.country == c], a.rec_sample) for i, d in ts.items() for c in sorted(d.country.unique())}).T.round(3)
        print(tp); tp.to_csv(f"{a.out}/test_profiles.csv")
        for c in sorted(set(ts[1].country) - set(s1.country)):
            for i, d in ts.items():
                print(f"\nunseen country {c}, S{i} sample:"); print(d[d.country == c].sample(8, random_state=0).to_string(index=False))


if __name__ == "__main__":
    main()