"""
extract_phi_condition.py — recover the two physically-meaningful attributes that the
PROVEDIt sample NAME encodes but the array extraction (prepare_data_set.py /
extract_metadata.py) drops:

  1. MIXTURE RATIO -> phi (mixture proportion per contributor).
     Mixture names: ...-<id1>_<id2>...-<r1;r2;...>-M<dil><cond>-<tpl>GF-Q...
       e.g. RD14-0003-31_32_33_34_35-1;1;2;9;1-M3a-0.21GF-...  -> phi = [1,1,2,9,1]/14
     Single-source: phi = one-hot on that donor.
     NOTE: this is the NOMINAL design ratio (template mixing ratio), not the realized
     per-locus proportion in the EPG (which degradation/dropout/amplification perturb).
     Useful as a target/covariate and to EVALUATE the phi head on real data.

  2. DNA CONDITION (degradation/UV/sonication/inhibition + level).
     Mixtures: in the M-block  -M<dil><code><level>   (a / b-e / S# / U# / I#).
     Single-source: in the d-block  <id>d<dil><code><level>.
       a        = untreated (pristine)
       b,c,d,e  = DNase I  (3,6,12,24 mU)
       S<n>     = sonication, n cycles
       U<n>     = UV exposure, n minutes
       I<n>     = humic-acid inhibition, n uL
       -15/-30/-45 = Fragmentase digestion minutes
     (Methods Table 1 + Sec 1, PROVEDIt naming convention.)

Reads data/meta_sample_names_{split}.json (saved aligned with the arrays by
prepare_data_set.py) -> no raw CSV re-prep. Writes, per split under data/:
  phi_{split}.npy        (N, 45) float32  nominal mixture proportion (cols = known_donors)
  condition_{split}.npy  (N,)    int32    condition category id (see COND_CATS)
  condition_level_{split}.npy (N,) float32 numeric level (mU / cycles / min / uL; NaN if n/a)
  condition_{split}.json          list[str] raw condition code per sample (audit)

Usage:  python extract_phi_condition.py            # all splits in data/
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = Path(__import__("os").environ.get("STR_DATA_DIR", str(ROOT / "data")))
META = json.load(open(DATA / "meta_set.json"))
KNOWN = META["known_donors"]
COL = {d: i for i, d in enumerate(KNOWN)}

COND_CATS = ["untreated", "dnase", "fragmentase", "sonication", "uv", "humic", "unknown"]
CAT_ID = {c: i for i, c in enumerate(COND_CATS)}
DNASE_MU = {"b": 3.0, "c": 6.0, "d": 12.0, "e": 24.0}


def parse_donors(name: str) -> list[int]:
    parts = name.split("-")
    if len(parts) < 3:
        return []
    contrib = parts[2]
    if "_" in contrib:
        out = []
        for seg in contrib.split("_"):
            m = re.match(r"^(\d+)", seg)
            if m:
                out.append(int(m.group(1)))
        return out
    m = re.match(r"^(\d+)", contrib)
    return [int(m.group(1))] if m else []


RE_RATIO = re.compile(r"-(\d+(?:[;:]\d+)+)-M")          # mixture ratio, anchored before M-block
RE_MBLOCK = re.compile(r"-M\d+([A-Za-z]*)(\d*)")        # mixture condition  -M<dil><code><level>
RE_DBLOCK = re.compile(r"d\d+([A-Za-z]+)(\d*)")         # single-source condition <id>d<dil><code><level>
RE_FRAG = re.compile(r"d\d+-(\d{2})-")                  # fragmentase single-source  d1-15-
RE_TEMPLATE = re.compile(r"-([\d.]+)(?:GF|IP|PP)\b")    # template mass ng  -0.21GF
RE_QINDEX = re.compile(r"-Q([\d.]+)")                   # PROVEDIt quality index -Q0.9


def _ffloat(m) -> float:
    try:
        return float(m.group(1)) if m else float("nan")
    except ValueError:
        return float("nan")


def parse_phi(name: str, donors: list[int]) -> np.ndarray:
    """Nominal mixture proportion -> (45,) on known-donor columns."""
    phi = np.zeros(45, dtype=np.float32)
    if len(donors) <= 1:
        if donors and donors[0] in COL:
            phi[COL[donors[0]]] = 1.0
        return phi
    m = RE_RATIO.search(name)
    if m:
        parts = re.split(r"[;:]", m.group(1))
        ratio = np.array([float(x) for x in parts], dtype=np.float64)
        if len(ratio) == len(donors) and ratio.sum() > 0:
            ratio = ratio / ratio.sum()
            for d, r in zip(donors, ratio):
                if d in COL:
                    phi[COL[d]] = r
            return phi
    # fallback: uniform over known contributors
    kd = [d for d in donors if d in COL]
    for d in kd:
        phi[COL[d]] = 1.0 / len(kd)
    return phi


def parse_condition(name: str, n_contrib: int) -> tuple[str, str, float]:
    """Return (category, raw_code, numeric_level)."""
    code, lvl = "", ""
    if n_contrib >= 2:
        m = RE_MBLOCK.search(name)
        if m:
            code, lvl = m.group(1), m.group(2)
    else:
        f = RE_FRAG.search(name)
        if f:
            return "fragmentase", f"-{f.group(1)}", float(f.group(1))
        m = RE_DBLOCK.search(name)
        if m:
            code, lvl = m.group(1), m.group(2)
    raw = f"{code}{lvl}"
    level = float(lvl) if lvl else float("nan")
    c = code[:1].lower() if code else ""
    if c == "a" or raw == "":
        return "untreated", raw or "a", float("nan")
    if c == "s":
        return "sonication", raw, level
    if c == "u":
        return "uv", raw, level
    if c == "i":
        return "humic", raw, level
    if c in DNASE_MU:
        return "dnase", raw, DNASE_MU[c]
    return "unknown", raw, level


def main():
    print(f"data dir: {DATA}")
    for split in ["train", "val", "test", "open", "dev"]:
        npath = DATA / f"meta_sample_names_{split}.json"
        if not npath.exists():
            continue
        names = json.load(open(npath))
        ymaybe = DATA / f"y_{split}_set.npy"
        y = np.load(ymaybe) if ymaybe.exists() else None
        N = len(names)
        phi = np.zeros((N, 45), np.float32)
        cat = np.zeros(N, np.int32)
        lvl = np.full(N, np.nan, np.float32)
        tpl = np.full(N, np.nan, np.float32)
        qix = np.full(N, np.nan, np.float32)
        raw_codes = []
        for i, nm in enumerate(names):
            s = str(nm)
            donors = parse_donors(s)
            phi[i] = parse_phi(s, donors)
            category, raw, level = parse_condition(s, len(donors))
            cat[i] = CAT_ID[category]; lvl[i] = level; raw_codes.append(raw)
            tpl[i] = _ffloat(RE_TEMPLATE.search(s)); qix[i] = _ffloat(RE_QINDEX.search(s))
        np.save(DATA / f"phi_{split}.npy", phi)
        np.save(DATA / f"condition_{split}.npy", cat)
        np.save(DATA / f"condition_level_{split}.npy", lvl)
        json.dump(raw_codes, open(DATA / f"condition_{split}.json", "w"))
        # template_ng / Q parsed from the SAME aligned names (extract_metadata.py's copies can be
        # misaligned with the arrays — different filtering); these supersede them for stratification.
        np.save(DATA / f"meta_template_{split}.npy", tpl)
        np.save(DATA / f"meta_qindex_{split}.npy", qix)
        # ── validation ───────────────────────────────────────────────────────
        rs = phi.sum(1)
        msg = f"  {split:5s} n={N:5d}  phi_sum[min={rs.min():.3f} max={rs.max():.3f}]"
        if y is not None:
            # phi must be nonzero exactly on the labelled donors
            agree = float(((phi > 0) == (y > 0)).all(1).mean())
            msg += f"  phi-support==y: {agree:.3f}"
        import collections
        cc = collections.Counter(COND_CATS[c] for c in cat)
        msg += "  cond=" + ",".join(f"{k}:{v}" for k, v in cc.most_common())
        print(msg)
    print("Done. phi_/condition_ saved to data/.")


if __name__ == "__main__":
    main()
