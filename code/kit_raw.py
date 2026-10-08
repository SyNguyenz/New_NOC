"""kit_raw.py - some PROVEDIt kits ship their GeneMapper exports as .xlsx (F6C: 5 of 6 Filtered files). Every reader in
the pipeline globs *.csv, so each .xlsx of the selected kit is written next to it as <name>.fromxlsx.csv (rewritten when
the .xlsx is newer). Run by preprocess.py before anything reads the raw files; the converted copies are git-ignored."""
import csv
import sys
from pathlib import Path

import openpyxl

import kit

HERE = Path(__file__).resolve().parent
# preprocess / calibrate read code/data_raw, make_insilico's design grid reads the project's data_raw: convert in both
RAW_ROOTS = [r for r in dict.fromkeys(p.resolve() for p in (HERE / "data_raw", HERE.parent / "data_raw")) if r.exists()]


def cell(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def main():
    n = 0
    for raw in RAW_ROOTS:
        for sub in ("PROVEDIt_1-5-Person CSVs Filtered", "PROVEDIt_1-5-Person CSVs UnFiltered"):
            for kd in (raw / sub).glob(f"*{kit.KIT}"):
                for f in sorted(kd.rglob("*.xlsx")):
                    if f.name.startswith("~$") or "Known Genotypes" in f.name:
                        continue
                    out = f.with_name(f.stem + ".fromxlsx.csv")
                    if out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
                        continue
                    ws = openpyxl.load_workbook(f, read_only=True, data_only=True).worksheets[0]
                    with open(out, "w", newline="", encoding="utf-8") as fh:
                        w = csv.writer(fh)
                        for row in ws.iter_rows(values_only=True):
                            w.writerow([cell(v) for v in row])
                    n += 1
    print(f"kit_raw: {kit.KIT}: {n} xlsx -> csv")


if __name__ == "__main__":
    sys.exit(main())
