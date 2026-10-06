import csv
import sys
from pathlib import Path

import openpyxl

import kit

RAW = Path(__file__).resolve().parent.parent / "data_raw"


def cell(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def main():
    n = 0
    for sub in ("PROVEDIt_1-5-Person CSVs Filtered", "PROVEDIt_1-5-Person CSVs UnFiltered"):
        for kd in (RAW / sub).glob(f"*{kit.KIT}"):
            for f in sorted(kd.rglob("*.xlsx")):
                out = f.with_suffix(".csv")
                if f.name.startswith("~$") or "Known Genotypes" in f.name or out.exists():
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
