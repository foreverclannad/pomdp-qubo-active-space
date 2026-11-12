"""Installed paper resources and the small, explicit public profile catalog."""

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE / "data"
CASE_NAMES = {
    "dio-s1min-dmrg": "NX1_DMRG",
    "dio-cioo-dmrg": "CI_OO_DMRG",
    "dio-s1min-cas": "NX1_CAS",
    "dio-cioo-cas": "CI_OO_CAS",
    "tif3": "TiF3",
}
# Generic input names reuse the existing numerical settings. Keep the five-case
# reproduction catalog separate so aliases never introduce extra paper cases.
PROFILE_NAMES = {**CASE_NAMES, "cas": "NX1_CAS", "dmrg": "NX1_DMRG"}
PROFILE_LABELS = {
    "dio-s1min-dmrg": "dio S1-min (DMRG)",
    "dio-cioo-dmrg": "dio CI-OO (DMRG)",
    "dio-s1min-cas": "dio S1-min (CAS)",
    "dio-cioo-cas": "dio CI-OO (CAS)",
    "tif3": "TiF3 (DMRG)",
    "cas": "New molecule (CAS; closed-shell starter)",
    "dmrg": "New molecule (DMRG; closed-shell starter)",
}
