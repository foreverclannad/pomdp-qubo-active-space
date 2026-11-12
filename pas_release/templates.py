"""Create small, editable inputs without reading cluster files or running solvers."""

import os
from pathlib import Path

from .common import read_json, require, write_new
from .inputs import SCHEMA
from .profiles import PAPER_INPUTS, paper_configuration
from .resources import CASE_NAMES, DATA, PROFILE_LABELS, PROFILE_NAMES


def catalog():
    return {
        name: dict(
            description=PROFILE_LABELS[name],
            initialization="original cluster seed" if name in CASE_NAMES else "editable H2 starter",
            settings_source=key,
        )
        for name, key in PROFILE_NAMES.items()
    }


def create(profile, output, *, seed_file=None, case_file=None):
    from .verify import validate_assets

    validate_assets()
    require(profile in PROFILE_NAMES, "Unknown template; run 'pas init --list' to see the choices")
    require(not (seed_file and case_file), "Choose either --seed-file or --case-file")
    case = read_json(DATA / "cases.json")["cases"][PROFILE_NAMES[profile]]
    config = paper_configuration(case)
    output = Path(output).expanduser().resolve()
    require(not output.exists(), f"Input already exists: {output}. Choose another --output path")
    companion = None
    if seed_file is not None or case_file is not None:
        field, selected = (
            ("seed_file", seed_file) if seed_file is not None else ("case_file", case_file)
        )
        selected = Path(selected).expanduser().resolve()
        require(selected != output, "--output must differ from the initialization file")
        try:
            selected = os.path.relpath(selected, output.parent)
        except ValueError:  # Different Windows drives cannot have a relative path.
            selected = str(selected)
        initial = {field: str(selected)}
    elif profile in CASE_NAMES:
        initial = {"seed_file": case["original_run"] + "/seed.json"}
    else:
        companion = output.with_name(output.stem + ".molecule.json")
        require(
            not companion.exists(),
            f"Molecular input already exists: {companion}. Use --case-file to reuse it, or choose another --output path",
        )
        initial = {"case_file": companion.name}
    value = dict(
        schema=SCHEMA,
        name=PROFILE_LABELS[profile],
        profile=profile,
        initialization=initial,
        evidence=dict(
            enabled=list(config["channels"]),
            weights={c: config["weights"][c] for c in config["channels"]},
            composition_alpha=config["composition_purity_weight"],
        ),
    )
    if profile in CASE_NAMES and seed_file is None and case_file is None:
        value.update(
            {field: binding["path"] for field, binding in PAPER_INPUTS[CASE_NAMES[profile]].items()}
        )
    if companion is not None:
        value["name"] = "H2 starter - replace with your molecule"
        molecule = dict(
            geometry_xyz="2\nH2, coordinates in angstrom\nH 0 0 0\nH 0 0 0.74\n",
            basis="STO-3G",
            charge=0,
            relativistic=None,
            group="NOSYM",
            active_space=dict(electrons=2, orbitals=2),
            states=dict(roots=[1, 2], weights=[0.5, 0.5], spin_multiplicity=1),
            target=dict(rules=[dict(elements=["H"])]),
        )
        write_new(companion, molecule)
        try:
            write_new(output, value)
        except OSError:
            companion.unlink()  # Roll back only the companion created in this call.
            raise
    else:
        write_new(output, value)
    return dict(
        input_file=str(output),
        profile=profile,
        created_files=[str(output)] + ([str(companion)] if companion else []),
        molecule_file=str(companion) if companion else None,
        initialization=initial,
        validated=False,
    )
