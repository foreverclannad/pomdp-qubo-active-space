"""Recompute the five paper valence controls and H2O symmetry comparison."""

import importlib.util
import sys

from ..common import read_json, require, sha
from ..resources import DATA as PAPER_DATA

DATA = PAPER_DATA / "small_molecules"


def run():
    if not __debug__:
        raise RuntimeError("Run validation without Python -O")
    import numpy as np
    import pyscf

    pyscf.lib.num_threads(1)
    provenance = read_json(DATA / "provenance.json")
    for name, digest in provenance["files"].items():
        if sha(DATA / name) != digest:
            raise ValueError("Changed small-molecule input: " + name)
    spec = importlib.util.spec_from_file_location(
        "paper_small_molecule_core", DATA / "control_core.py"
    )
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    cases = {}
    water = None
    table = {
        "H2": "2.0000",
        "HF": "4.9949",
        "H2O": "5.9887",
        "NH3": "6.9775",
        "O2_pi_pi_star": "4.0000",
    }
    for key, filename in [
        ("H2", "h2"),
        ("HF", "hf"),
        ("H2O", "h2o"),
        ("NH3", "nh3"),
        ("O2_pi_pi_star", "o2_pi_pi_star"),
    ]:
        payload = read_json(DATA / "inputs" / f"{filename}.json")
        saved = read_json(DATA / "results" / key / "result.json")
        result = core.run_case(payload, use_symmetry=False)
        fresh = result.result
        require(fresh["passed"], (key, fresh["checks"]))
        a, b = (fresh["selected_active_space"], saved["selected_active_space"])
        require(a["indices"] == b["indices"], key)
        require(a["occupation_bins"] == b["occupation_bins"], key)
        require(fresh["active_space_target"] == saved["active_space_target"], key)
        delta = fresh["molecule"]["scf_energy_hartree"] - saved["molecule"]["scf_energy_hartree"]
        pdiff = a["projection_score"] - b["projection_score"]
        require(abs(delta) < 1e-08, (key, delta))
        require(abs(pdiff) < 1e-07, (key, pdiff))
        require(format(a["projection_score"], ".4f") == table[key], key)
        cases[key] = dict(
            result=fresh,
            energy_difference_hartree=delta,
            projection_difference=pdiff,
            reported_projection_sum=table[key],
            paper_match=True,
        )
        if key == "H2O":
            water = result
        print(key + ": passed", file=sys.stderr, flush=True)
    payload = read_json(DATA / "inputs/h2o.json")
    symmetric = core.run_case(payload, use_symmetry=True)
    comparison = core.compare_subspaces(
        water, symmetric, energy_tolerance=1e-08, minimum_singular_value=0.999999
    )
    require(comparison["passed"], "H2O C1/C2v subspace comparison failed")
    require(
        all((format(x, ".4f") == "1.0000" for x in comparison["principal_singular_values"])),
        "H2O symmetry singular values no longer round to 1.0000",
    )
    require(
        format(symmetric.result["molecule"]["scf_energy_hartree"], ".6f") == "-76.009132",
        "H2O symmetric SCF energy no longer rounds to -76.009132 hartree",
    )
    require(
        all((sha(DATA / n) == v for n, v in provenance["files"].items())),
        "Small-molecule reference material changed during verification",
    )
    return dict(
        schema="pas_paper_small_molecule_verification_v1",
        source=provenance,
        scope="Recomputed paper valence construction and symmetry; not historical physical-feedback trajectories",
        python=sys.version,
        numpy=np.__version__,
        pyscf=pyscf.__version__,
        archived_pyscf="2.11.0",
        fresh_SCF_calculations=6,
        new_OpenMolcas_Block2_jobs=0,
        tolerances=dict(energy_hartree=1e-08, projection_sum=1e-07),
        cases=cases,
        H2O_C1_C2v=comparison,
        inputs_unchanged=True,
        passed=True,
    )
