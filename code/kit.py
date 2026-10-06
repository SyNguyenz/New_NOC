import os

KIT = os.environ.get("STR_KIT", "3500_GF29cycles")
_SPEC = {
    "3500_GF29cycles": dict(tag="GF", geno="PROVEDIt_RD14-0003 GF Known Genotypes.xlsx"),
    "3500_F6C29cycles_hlfrxn": dict(tag="F6C", geno="PROVEDIt_RD14-0003 F6C Known Genotypes.csv"),
}
if KIT not in _SPEC:
    raise SystemExit(f"STR_KIT={KIT!r} not in {sorted(_SPEC)}")
TAG = _SPEC[KIT]["tag"]
GENO_FILE = _SPEC[KIT]["geno"]
REPEAT_BP = {"D22S1045": 3, "Penta D": 5, "Penta E": 5}
DYE = {
    "3500_GF29cycles": {"D3S1358": 0, "vWA": 0, "D16S539": 0, "CSF1PO": 0, "TPOX": 0,
                        "Yindel": 1, "AMEL": 1, "D8S1179": 1, "D21S11": 1, "D18S51": 1,
                        "DYS391": 2, "D2S441": 2, "D19S433": 2, "TH01": 2, "FGA": 2,
                        "D22S1045": 3, "D5S818": 3, "D13S317": 3, "D7S820": 3, "SE33": 3,
                        "D10S1248": 4, "D1S1656": 4, "D12S391": 4, "D2S1338": 4},
    "3500_F6C29cycles_hlfrxn": {"AMEL": 0, "D3S1358": 0, "D1S1656": 0, "D2S441": 0, "D10S1248": 0, "D13S317": 0,
                                "Penta E": 0, "D16S539": 1, "D18S51": 1, "D2S1338": 1, "CSF1PO": 1, "Penta D": 1,
                                "TH01": 2, "vWA": 2, "D21S11": 2, "D7S820": 2, "D5S818": 2, "TPOX": 2,
                                "D8S1179": 3, "D12S391": 3, "D19S433": 3, "SE33": 3, "D22S1045": 3,
                                "DYS391": 4, "FGA": 4, "DYS576": 4, "DYS570": 4},
}[KIT]


def unit_scale(locus_name):
    return 5 if REPEAT_BP.get(locus_name) == 5 else 4
