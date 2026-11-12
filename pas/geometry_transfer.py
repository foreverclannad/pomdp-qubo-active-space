"""Explicit geometry changes, separate from same-geometry response sharing.

Only orbitals and the SCF role ledger travel between points. The target-native
optimizer orthonormalizes the inherited INPORB in its new AO metric; no old
quality observation is relabelled as an observation of the new Hamiltonian.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import read_json
from .orbital_io import read_openmolcas_orbital_hdf5


def xyz_atoms(text):
    rows = text.splitlines()
    atoms = [line.split() for line in rows[2:] if line.strip()]
    if len(atoms) != int(rows[0]) or not atoms or any(len(row) != 4 for row in atoms):
        raise ValueError("Malformed sequence geometry")
    coordinates = np.asarray([row[1:] for row in atoms], float)
    if not np.isfinite(coordinates).all() or rows[1].strip().lower() == "bohr":
        raise ValueError("Sequence manifest requires finite Angstrom XYZ coordinates")
    return [row[0] for row in atoms], coordinates


def validate_change(parent_case, child_case):
    from .cold_start import physical_case

    old, new = physical_case(parent_case), physical_case(child_case)
    for name in ("key", "case_id", "geometry_xyz"):
        old.pop(name, None)
        new.pop(name, None)
    if old != new:
        raise ValueError(
            "Geometry transfer may not change basis, Hamiltonian, states, target or CAS"
        )
    if xyz_atoms(parent_case["geometry_xyz"])[0] != xyz_atoms(child_case["geometry_xyz"])[0]:
        raise ValueError("Geometry transfer changed atom identities/order")


def ao_order(mo):
    # PySCF real spherical p functions use x,y,z; higher l uses m=-l,...,+l.
    def rank(angular_momentum, m):
        if angular_momentum == 1:
            return {1: 0, -1: 1, 0: 2}[m]
        if not 0 <= angular_momentum <= 4 or not -angular_momentum <= m <= angular_momentum:
            raise ValueError("Unsupported spherical AO component")
        return m + angular_momentum

    keys = [
        (int(c), int(angular_momentum), int(r), rank(int(angular_momentum), int(m)))
        for c, r, angular_momentum, m in mo.basis_function_ids
    ]
    if len(set(keys)) != mo.orbital_count or len(mo.symmetry_block_sizes) != 1:
        raise ValueError("Ambiguous or symmetry-adapted AO representation")
    return np.asarray(sorted(range(len(keys)), key=keys.__getitem__))


def molecule(basis, molden_path):
    from pyscf.tools import molden

    mo = read_openmolcas_orbital_hdf5(basis)
    mol, _, coeff, _, _, _ = molden.load(str(molden_path))
    order = ao_order(mo)
    metric = mol.intor_symmetric("int1e_ovlp")
    c = mo.mo_coefficients[order]
    with h5py.File(basis) as h:
        xyz = np.asarray(h["CENTER_COORDINATES"][()], float)
    labels = [re.match(r"[A-Za-z]+", s).group(0).lower() for s in mo.center_labels]
    if (
        mol.cart
        or labels != [mol.atom_pure_symbol(i).lower() for i in range(mol.natm)]
        or np.shape(coeff) != c.shape
        or xyz.shape != mol.atom_coords().shape
        or not np.allclose(xyz, mol.atom_coords(), atol=2e-7, rtol=0)
        or not np.allclose(metric, mo.ao_overlap[np.ix_(order, order)], atol=2e-7, rtol=0)
        or not np.allclose(coeff, c, atol=2e-6, rtol=2e-7)
    ):
        raise ValueError("Molden/HDF5 atom, AO metric or orbital binding differs")
    if not np.allclose(c.T @ metric @ c, np.eye(mo.orbital_count), atol=2e-6, rtol=0):
        raise ValueError("MOs are not orthonormal in their own geometry")
    gram = c.T @ metric @ c
    values, vectors = np.linalg.eigh((gram + gram.T) / 2)
    c = c @ (vectors / np.sqrt(values)) @ vectors.T
    return mol, c, metric, mo


def cross_data(left_basis, left_molden, right_basis, right_molden):
    from pyscf import gto

    left, right = molecule(left_basis, left_molden), molecule(right_basis, right_molden)
    if not np.array_equal(left[0].atom_charges(), right[0].atom_charges()) or left[0].ao_labels(
        fmt=False
    ) != right[0].ao_labels(fmt=False):
        raise ValueError("Cross-geometry AO identities changed")
    cross = np.asarray(gto.intor_cross("int1e_ovlp", left[0], right[0]), float)
    overlap = left[1].T @ cross @ right[1]
    if not np.isfinite(overlap).all() or np.max(np.abs(overlap)) > 1 + 1e-6:
        raise ValueError("Invalid cross-geometry MO overlap")
    return left, right, cross, overlap


def bound_overlap(old_binding, new_binding, evidence):
    from .backend import verify_binding

    if (
        set(evidence) != {"schema", "parent_molden", "child_molden"}
        or evidence["schema"] != "pas_cross_geometry_ao_overlap_v1"
    ):
        raise ValueError("Invalid geometry-overlap provenance")
    _, _, _, overlap = cross_data(
        verify_binding(old_binding),
        verify_binding(evidence["parent_molden"]),
        verify_binding(new_binding),
        verify_binding(evidence["child_molden"]),
    )
    return overlap


def locate_molden(basis):
    basis = Path(basis)
    name = basis.with_suffix(".molden").name
    for path in (basis.with_suffix(".molden"), basis.parent / "work/job" / name):
        if path.is_file():
            molecule(basis, path)
            return path
    raise ValueError("Missing bound native Molden file: " + str(basis))


def output_context(parent, partition, basis, checkpoint, config, transition):
    from . import backend, electron_roles

    case = transition["child_case"]
    validate_change(parent.payload["case"], case)
    parsed = backend.context_from_files(case, basis, checkpoint, config, electrons=parent.electrons)
    proof = electron_roles.from_geometry_context(
        parent,
        partition,
        backend.binding(basis),
        electron_roles.partition(parsed),
        transition["parent_molden"],
        backend.binding(locate_molden(basis)),
    )[1]
    return backend.context_from_files(
        case, basis, checkpoint, config, electrons=parent.electrons, role_provenance=proof
    )


def audit_output(parent, output, partition, transition, directory):
    from . import backend

    validate_change(parent.payload["case"], output.payload["case"])
    for field in (
        "backend",
        "nmo",
        "total_electrons",
        "electrons",
        "states",
        "protocol_id",
        "effective_config_id",
    ):
        if getattr(parent, field) != getattr(output, field):
            raise ValueError("Geometry step changed " + field)
    for group in ("active", "inactive", "frozen", "deleted"):
        if len(getattr(output, group)) != len(partition[group]):
            raise ValueError("Geometry step changed " + group)
    basis = Path(output.payload["basis_file"])
    if (Path(directory) / "canonical_gauge/gauge.json").exists():
        from pyscf.tools import molden

        native = backend.verify_binding(
            read_json(Path(directory) / "canonical_gauge/gauge.json")["source_basis"]
        )
        mol, _, _, _ = molecule(native, locate_molden(native))
        mo = read_openmolcas_orbital_hdf5(basis)
        dest = basis.with_suffix(".molden")
        if not dest.exists():
            molden.from_mo(
                mol,
                str(dest),
                mo.mo_coefficients[ao_order(mo)],
                ene=mo.energies,
                occ=mo.occupations,
            )
    evidence = dict(
        schema="pas_cross_geometry_ao_overlap_v1",
        parent_molden=transition["parent_molden"],
        child_molden=backend.binding(locate_molden(basis)),
    )
    cross = bound_overlap(
        backend.binding(parent.payload["basis_file"]), backend.binding(basis), evidence
    )
    rows, cols = linear_sum_assignment(-(np.abs(cross) ** 2))
    mapping = dict(zip(rows.tolist(), cols.tolist()))
    output.stable_order = tuple(mapping[i] for i in parent.stable_order)
    return dict(
        model="cross_geometry_ao_overlap_v1",
        old_to_new=[mapping[i] for i in range(parent.nmo)],
        overlap_evidence=evidence,
        response_posterior_transported=False,
        full_span_semantics="each_frame_complete_in_its_own_AO_metric",
    )
