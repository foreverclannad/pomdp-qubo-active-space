"""Small shared input fixture; no solver is executed."""

import json

import pytest


@pytest.fixture
def input_file(tmp_path):
    molecule = dict(
        geometry_xyz="2\nH2\nH 0 0 0\nH 0 0 0.74\n",
        basis="STO-3G",
        charge=0,
        relativistic=None,
        group="NOSYM",
        active_space=dict(electrons=2, orbitals=2),
        states=dict(roots=[1, 2], weights=[0.5, 0.5], spin_multiplicity=1),
        target=dict(rules=[dict(elements=["H"])]),
    )
    (tmp_path / "molecule.json").write_text(json.dumps(molecule))
    value = dict(
        schema="pas.input.v1",
        name="H2 input example",
        profile="dio-s1min-cas",
        initialization=dict(case_file="molecule.json"),
    )
    path = tmp_path / "input.json"
    path.write_text(json.dumps(value))
    return path
