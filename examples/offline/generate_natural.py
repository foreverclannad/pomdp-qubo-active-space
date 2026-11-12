"""Regenerate the H4 natural-orbital example using optional PySCF.

Run after installing PAS, with --output pointing to a new directory. The
distributed example already contains all arrays and needs no chemistry solver.
"""

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyscf
from generate import orbital_entropy
from pyscf import fci, gto, lib, scf

from pas_release import offline


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def generate(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    lib.num_threads(1)
    mol = gto.M(atom="H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    assert mf.converged
    solver = fci.FCI(mf)
    solver.conv_tol = 1e-13
    energy, ci = solver.kernel()
    sites = mol.nao_nr()
    density = solver.make_rdm1(ci, sites, mol.nelec)
    occupations, rotation = np.linalg.eigh(density)
    occupations, rotation = occupations[::-1], rotation[:, ::-1]
    coefficients = mf.mo_coeff @ rotation
    signs = np.sign(coefficients[np.argmax(abs(coefficients), axis=0), np.arange(sites)])
    coefficients *= signs

    # Recompute the FCI state and QIT in these natural orbitals. Entropies from
    # the original RHF orbital basis cannot be reused after an orbital rotation.
    natural_solver = fci.FCI(mol, coefficients)
    natural_solver.conv_tol = 1e-13
    natural_energy, natural_ci = natural_solver.kernel()
    natural_density = natural_solver.make_rdm1(natural_ci, sites, mol.nelec)
    diagonal_error = float(np.max(abs(natural_density - np.diag(occupations))))
    ao_error = float(
        np.max(
            abs(
                coefficients @ natural_density @ coefficients.T
                - mf.mo_coeff @ density @ mf.mo_coeff.T
            )
        )
    )
    assert diagonal_error < 1e-10 and ao_error < 1e-10
    assert abs(natural_energy - energy) < 1e-10
    entropy = np.array([orbital_entropy(natural_ci, sites, mol.nelec, [i]) for i in range(sites)])
    mutual = np.zeros((sites, sites))
    for i, j in itertools.combinations(range(sites), 2):
        pair_entropy = orbital_entropy(natural_ci, sites, mol.nelec, [i, j])
        mutual[i, j] = mutual[j, i] = 0.5 * (entropy[i] + entropy[j] - pair_entropy)
    np.savez_compressed(
        output / "natural-orbitals.npz",
        coefficients=coefficients,
        overlap=mf.get_ovlp(),
        ao_centers=np.arange(1, 5),
        ao_angular_momenta=np.zeros(4, int),
        center_labels=np.array(["H"] * 4),
        coordinates_angstrom=mol.atom_coords(unit="Angstrom"),
    )
    qit = dict(single_orbital_entropy=entropy[None], mutual_information=mutual[None])
    np.savez_compressed(output / "natural-qit.npz", **qit)
    np.savez_compressed(output / "natural-observations.npz", densities=natural_density[None], **qit)
    raw = dict(
        schema="pas.offline.v1",
        name="H4 / STO-3G / full-CI natural occupations",
        source=dict(
            program="PySCF",
            version=pyscf.__version__,
            method="Singlet full-CI natural orbitals and occupations",
            description="All evidence uses the same FCI natural-orbital basis",
        ),
        orbitals=dict(format="npz", file="natural-orbitals.npz"),
        observations=dict(
            file="natural-qit.npz",
            orbitals=[1, 2, 3, 4],
            electrons=4,
            orbital_file_sha256=hashlib.sha256(
                (output / "natural-orbitals.npz").read_bytes()
            ).hexdigest(),
            density_convention="spin_summed_natural_orbital",
            natural_occupations=[occupations.tolist()],
            entropy_unit="natural_log",
            mutual_information_convention="half_mutual_information",
        ),
        selection=dict(
            electrons=2, orbitals=2, reference_occupations=[2, 2, 0, 0], initial_active=[1, 3]
        ),
        states=dict(ids=["singlet_ground_state"], weights=[1.0], spin_multiplicity=1),
        evidence=dict(
            enabled=["occupation", "composition", "qit"],
            weights=dict(occupation=1.0, composition=1.0, qit=1.0),
            composition_alpha=0.45,
        ),
        target=dict(atoms=[1, 2, 3, 4], angular_momenta=[0]),
    )
    write_json(output / "input-natural.json", raw)
    reference = copy.deepcopy(raw)
    reference["observations"].pop("natural_occupations")
    reference["observations"].update(
        file="natural-observations.npz", density_convention="spin_summed_mo"
    )
    write_json(output / "input-natural-density.json", reference)
    minimal = copy.deepcopy(raw)
    minimal["evidence"] = dict(enabled=["occupation"])
    minimal.pop("target")
    for field in ("file", "entropy_unit", "mutual_information_convention"):
        minimal["observations"].pop(field)
    write_json(output / "input-natural-occupation.json", minimal)

    expected = dict(tolerance=1e-10)
    with TemporaryDirectory(prefix="pas-natural-example-") as temporary:
        for label, name in (
            ("all_evidence", "input-natural.json"),
            ("occupation_only", "input-natural-occupation.json"),
        ):
            result = offline.select(output / name, Path(temporary) / label)
            expected[label] = {
                key: result[key] for key in ("initial", "recommended", "candidates", "score_change")
            }
    write_json(output / "expected-natural.json", expected)
    write_json(
        output / "natural-provenance.json",
        dict(
            source_program="PySCF",
            pyscf=pyscf.__version__,
            numpy=np.__version__,
            method="FCI in RHF orbitals, density diagonalization, then FCI in natural orbitals",
            basis="STO-3G",
            orbital_order="Descending natural occupation; largest-magnitude AO coefficient positive",
            source_energy_hartree=float(energy),
            natural_basis_energy_hartree=float(natural_energy),
            natural_density_diagonal_max_error=diagonal_error,
            ao_density_invariance_max_error=ao_error,
            density_trace=float(np.trace(natural_density)),
            fci_norm=float(np.linalg.norm(natural_ci)),
            entropy_method="Exact FCI Fock-space partial trace in the natural-orbital basis",
            file_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(output.iterdir())
                if p.is_file()
            },
        ),
    )
    print(
        json.dumps(
            dict(
                output=str(output),
                natural_occupations=occupations.tolist(),
                expected=expected["all_evidence"]["recommended"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    generate(parser.parse_args().output)
