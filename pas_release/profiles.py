"""Paper run defaults and companion reference inputs."""

from pas.config import configuration

from .common import read_json, require
from .resources import DATA

PAPER_INPUTS = read_json(DATA / "paper_inputs.json")


def paper_configuration(case):
    return configuration(
        {**case["config"], "composition_purity_weight": case["paper_composition_purity_weight"]}
    )


def check_paper_materials(case_name, files):
    """An original paper seed must carry the original evidence and stopping inputs."""
    for field, original in PAPER_INPUTS[case_name].items():
        require(
            field in files,
            f"{field}: required with this paper seed; regenerate the complete input with 'pas init'",
        )
        require(
            files[field]["sha256"] == original["sha256"],
            f"{field}: differs from the original paper material; use a generic cas/dmrg profile for a different experiment",
        )
