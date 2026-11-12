"""Export paper tables and molecular inputs from a paper reproduction run."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pas import acceptance
from pas.contracts import Context
from pas_release.common import Evidence, read_json, require, sha
from pas_release.resources import CASE_NAMES, DATA


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(run, evidence_directory, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    require(not output.exists(), "Choose a new export directory: " + str(output))
    require(read_json(run / "validation/summary.json")["passed"], "paper verification did not pass")
    evidence = Evidence(evidence_directory)
    evidence.verify_all()
    results = read_json(run / "result.json")["cases"]
    cases = read_json(DATA / "cases.json")["cases"]
    physical = {}
    for group in ("dio", "native"):
        physical.update(read_json(run / f"validation/physical_{group}.json")["runs"])
    spaces, energies, gaps, occupations, inputs = [], [], [], [], {}
    subspaces = []
    for public_name, key in CASE_NAMES.items():
        case, result = cases[key], results[public_name]
        root = Path(case["original_run"])
        seed, state = (evidence.read(root / (name + ".json")) for name in ("seed", "state"))
        context = Context.from_dict(seed["context"])
        inputs[public_name] = context.payload["case"]
        checks = result["reference_checks"]
        spaces.append(
            dict(
                example=public_name,
                electrons=context.electrons,
                orbitals=context.k,
                active_indices=" ".join(map(str, result["active_indices"])),
                basis_sha256=Context.from_dict(state["incumbent"]["output_context"]).basis_id,
                occupation=result["metrics"]["occupation"],
                composition=result["metrics"]["composition"],
                qit=result["metrics"].get("qit", ""),
                quality=result["metrics"]["quality"],
                average_energy_hartree=result["metrics"]["energy"],
                reference_coverage="; ".join(
                    f"{name}: {row['directions']}/{row['dimension']}"
                    for name, row in checks.items()
                ),
                candidate_optimizations=case["optimization_choices"],
            )
        )
        for phase, receipt in (("initial", seed["incumbent"]), ("final", state["incumbent"])):
            ctx = Context.from_dict(receipt["output_context"])
            endpoint = physical[key]["rows"][0] if phase == "initial" else result
            roots = endpoint["root_energies_hartree"]
            ground = roots[0]
            for i, energy in enumerate(roots):
                energies.append(
                    dict(
                        example=public_name,
                        phase=phase,
                        root=ctx.states["roots"][i],
                        state=ctx.states["ids"][i],
                        weight=ctx.states["weights"][i],
                        energy_hartree=energy,
                        excitation_eV=(energy - ground) * 27.211386245988,
                        excitation_cm_inverse=(energy - ground) * 219474.6313705,
                    )
                )
            groups = ((1, 2), (3, 4)) if key == "TiF3" else ((1,),)
            for group in groups:
                gap = sum(roots[i] - ground for i in group) / len(group)
                gaps.append(
                    dict(
                        example=public_name,
                        phase=phase,
                        reference_root=ctx.states["roots"][0],
                        upper_roots=" ".join(str(ctx.states["roots"][i]) for i in group),
                        excitation_eV=gap * 27.211386245988,
                        excitation_cm_inverse=gap * 219474.6313705,
                    )
                )
            for label in checks:
                reference = state["acceptance"]["comparisons"][label]
                compare = {
                    acceptance.RASORB_COMPARATOR_ID: acceptance.compare_hdf5_to_rasorb,
                    acceptance.HDF5_COMPARATOR_ID: acceptance.compare_hdf5_active_subspaces,
                }[reference["comparator_id"]]
                subspace = compare(
                    evidence.bind(ctx.payload["basis_file"], ctx.basis_id),
                    evidence.artifact(reference["reference"]),
                    expected_active_orbitals=ctx.k,
                )["subspace"]
                subspaces.append(
                    dict(
                        example=public_name,
                        phase=phase,
                        reference=label,
                        coverage=subspace["directions_at_or_above_0_95"],
                        dimension=subspace["dimension"],
                        minimum_singular_value=subspace["minimum_singular_value"],
                    )
                )
            density_file = evidence.artifact(receipt["artifacts"]["densities"])
            density_hash = sha(density_file)
            density = np.load(density_file, allow_pickle=False)
            require(
                density.shape == (len(ctx.states["roots"]), ctx.k, ctx.k),
                "Density shape does not match the orbital/state scope",
            )
            for r, root_id in enumerate(ctx.states["roots"]):
                for position, mo in enumerate(ctx.active):
                    occupations.append(
                        dict(
                            example=public_name,
                            phase=phase,
                            root=root_id,
                            state=ctx.states["ids"][r],
                            active_position=position + 1,
                            mo_index=mo + 1,
                            occupation=float(density[r, position, position]),
                            basis_sha256=ctx.basis_id,
                            density_sha256=density_hash,
                        )
                    )
    small = read_json(run / "validation/small_molecules.json")
    small_rows = []
    for name, record in small["cases"].items():
        result = record["result"]
        active = result["selected_active_space"]
        small_rows.append(
            dict(
                molecule=name,
                electrons=result["active_space_target"]["electrons"],
                orbitals=result["active_space_target"]["orbitals"],
                active_indices=" ".join(str(i + 1) for i in active["indices"]),
                projection_sum=active["projection_score"],
                scf_energy_hartree=result["molecule"]["scf_energy_hartree"],
                doubly_occupied=active["occupation_bins"]["doubly_occupied"],
                singly_occupied=active["occupation_bins"]["singly_occupied"],
                empty=active["occupation_bins"]["empty"],
            )
        )
    controls = read_json(run / "validation/controls.json")
    subspace = controls["CI_CC"]["comparison"]
    for phase in ("initial", "final"):
        subspaces.append(
            dict(
                example="dio-cicc",
                phase=phase,
                reference="reference_subspace",
                coverage=subspace["directions_at_or_above_0_95"],
                dimension=subspace["dimension"],
                minimum_singular_value=subspace["minimum_singular_value"],
            )
        )
    evidence.unchanged()
    output.mkdir(parents=True)
    write_csv(output / "active_spaces.csv", spaces)
    write_csv(output / "root_energies.csv", energies)
    write_csv(output / "excitation_energies.csv", gaps)
    write_csv(output / "reference_subspaces.csv", subspaces)
    write_csv(
        output / "decision_control.csv",
        [
            dict(
                candidate=name,
                coverage=row["coverage"],
                quality=row["metrics"]["quality"],
                average_energy_hartree=row["metrics"]["energy"],
                delta_average_energy_mEh=row["delta_average_energy_mEh"],
            )
            for name, row in controls["controls"].items()
        ],
    )
    write_csv(output / "small_molecules.csv", small_rows)
    write_csv(output / "orbital_occupations.csv", occupations)
    (output / "molecular_inputs").mkdir()
    for name, case in inputs.items():
        (output / "molecular_inputs" / (name + ".json")).write_text(
            json.dumps(case, indent=2) + "\n", encoding="utf-8"
        )
        if "geometry_xyz" in case:
            (output / "molecular_inputs" / (name + ".xyz")).write_text(
                case["geometry_xyz"], encoding="utf-8"
            )
    for filename, value in (
        ("controls.json", controls),
        ("water_symmetry.json", small["H2O_C1_C2v"]),
    ):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(
        f"Exported {len(spaces)} main cases and {len(small_rows)} small-molecule cases to {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.evidence, args.output)
