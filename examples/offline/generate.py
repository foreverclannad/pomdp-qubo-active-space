"""Regenerate the H4/STO-3G full-CI reviewer data (optional PySCF + MOKIT).

The published example already contains all data. This script performs a fresh
RHF/FCI calculation only when explicitly invoked with a new output directory.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from pyscf import fci, gto, lib, scf


def orbital_entropy(ci, sites, nelec, selected):
    """Trace the exact FCI state after a fermionic permutation of spatial modes."""
    other = [i for i in range(sites) if i not in selected]
    order = [mode for i in [*selected, *other] for mode in (i, sites + i)]
    positions = {mode: i for i, mode in enumerate(order)}
    wave = np.zeros((4 ** len(selected), 4 ** len(other)))
    strings = [fci.cistring.make_strings(range(sites), n) for n in nelec]
    for ia, a in enumerate(strings[0]):
        for ib, b in enumerate(strings[1]):
            occupied = [i for i in range(sites) if a >> i & 1]
            occupied += [sites + i for i in range(sites) if b >> i & 1]
            permutation = [positions[i] for i in occupied]
            parity = sum(x > y for x, y in itertools.combinations(permutation, 2)) % 2
            digits = [int(a >> i & 1) + 2 * int(b >> i & 1) for i in range(sites)]
            left = sum(digits[i] * 4**j for j, i in enumerate(selected))
            right = sum(digits[i] * 4**j for j, i in enumerate(other))
            wave[left, right] = (-1) ** parity * ci[ia, ib]
    probabilities = np.linalg.svd(wave, compute_uv=False) ** 2
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities)))


def generate(output):
    import mokit
    import pyscf
    from mokit.lib.py2fch import write_pyscf_dm_into_fch
    from mokit.lib.py2fch_direct import fchk

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    lib.num_threads(1)
    mol = gto.M(atom="H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6", basis="sto-3g", spin=0, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    assert mf.converged
    solver = fci.FCI(mf)
    solver.conv_tol = 1e-13
    energy, ci = solver.kernel()
    sites = mol.nao_nr()
    density = solver.make_rdm1(ci, sites, mol.nelec)
    entropy = np.array([orbital_entropy(ci, sites, mol.nelec, [i]) for i in range(sites)])
    mutual = np.zeros((sites, sites))
    for i, j in itertools.combinations(range(sites), 2):
        pair_entropy = orbital_entropy(ci, sites, mol.nelec, [i, j])
        mutual[i, j] = mutual[j, i] = 0.5 * (entropy[i] + entropy[j] - pair_entropy)
    np.savez_compressed(
        output / "orbitals.npz",
        coefficients=mf.mo_coeff,
        overlap=mf.get_ovlp(),
        ao_centers=np.arange(1, 5),
        ao_angular_momenta=np.zeros(4, int),
        center_labels=np.array(["H"] * 4),
        coordinates_angstrom=mol.atom_coords(unit="Angstrom"),
    )
    np.savez_compressed(
        output / "observations.npz",
        densities=density[None],
        single_orbital_entropy=entropy[None],
        mutual_information=mutual[None],
    )
    np.savez_compressed(
        output / "qit.npz", single_orbital_entropy=entropy[None], mutual_information=mutual[None]
    )
    fch = str(output / "orbitals.fch")
    fchk(mf, fch, density=True)
    ao_density = mf.mo_coeff @ density @ mf.mo_coeff.T
    # As in MOKIT's correlated *_NO.fch exports, this named section carries
    # the correlated total density; source.method explicitly identifies FCI.
    write_pyscf_dm_into_fch(fch, 1, sites, ao_density, False)
    raw = dict(
        schema="pas.offline.v1",
        name="H4 / STO-3G / full-CI imported observations",
        source=dict(
            program="PySCF",
            version=pyscf.__version__,
            method="RHF orbitals and full-CI singlet",
            description="Fixed exported data; selecting CAS(2e,2o) from a four-electron, four-orbital observation",
        ),
        orbitals=dict(format="npz", file="orbitals.npz"),
        observations=dict(
            file="observations.npz",
            orbitals=[1, 2, 3, 4],
            electrons=4,
            orbital_file_sha256=hashlib.sha256((output / "orbitals.npz").read_bytes()).hexdigest(),
            density_convention="spin_summed_mo",
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
    (output / "input.json").write_text(json.dumps(raw, indent=2) + "\n")
    raw["orbitals"] = dict(format="mokit_fch", file="orbitals.fch")
    raw["observations"].update(
        file="qit.npz",
        fch_density="Total SCF Density",
        orbital_file_sha256=hashlib.sha256((output / "orbitals.fch").read_bytes()).hexdigest(),
    )
    (output / "input-mokit.json").write_text(json.dumps(raw, indent=2) + "\n")
    provenance = dict(
        molecule=mol.dumps(),
        source_energy_hartree=float(energy),
        scf_energy_hartree=float(mf.e_tot),
        natural_occupations=np.linalg.eigvalsh(density)[::-1].tolist(),
        density_trace=float(np.trace(density)),
        fci_norm=float(np.linalg.norm(ci)),
        source_program="PySCF",
        pyscf=pyscf.__version__,
        numpy=np.__version__,
        mokit=getattr(mokit, "__version__", "unknown"),
        entropy_method="Exact FCI Fock-space partial trace after fermionic permutation",
        file_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.iterdir())
            if p.is_file()
        },
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(output),
                energy_hartree=float(energy),
                natural_occupations=provenance["natural_occupations"],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    generate(parser.parse_args().output)
