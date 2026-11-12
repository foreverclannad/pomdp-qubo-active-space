"""Verified cross-basis projector features and exact quadratic GP prior draws.

All frames share one physical AO metric and Hamiltonian.  Orbital numbers are
local selection variables, never identifiers for transporting observations.
"""

from __future__ import annotations

import copy
from pathlib import Path

import h5py
import numpy as np

from . import electron_roles
from .common import file_hash, identity
from .contracts import Context
from .orbital_io import read_openmolcas_orbital_hdf5


def _real(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(name + " must be real; complex orbital frames are unsupported")
    value = np.asarray(raw, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError(name + " must be finite")
    return value


def _load_native(context, path):
    case = context.payload.get("case")
    required = ("geometry_xyz", "basis", "charge", "relativistic", "group")
    if not isinstance(case, dict) or any(key not in case for key in required):
        raise ValueError("Native projector frame lacks geometry/Hamiltonian metadata")
    if identity(case["geometry_xyz"]) != context.geometry_id:
        raise ValueError("Projector frame geometry identity differs from its physical case")
    with h5py.File(path, "r") as handle:
        for name in ("MO_VECTORS", "AO_OVERLAP_MATRIX"):
            _real(handle[name][()], name)
        if "CENTER_COORDINATES" not in handle:
            raise ValueError("Native projector frame lacks AO centre coordinates")
        coordinates = _real(handle["CENTER_COORDINATES"][()], "AO centre coordinates")
    mo = read_openmolcas_orbital_hdf5(path)
    if len(mo.symmetry_block_sizes) != 1:
        raise ValueError("Native projector frame currently requires the NOSYM AO layout")
    ids = np.asarray(mo.basis_function_ids)
    if len({tuple(row) for row in ids.tolist()}) != mo.orbital_count:
        raise ValueError("Native AO identities are ambiguous")
    if not mo.center_labels or any(not label for label in mo.center_labels):
        raise ValueError("Native projector frame lacks AO centre identities")
    from .backend import _geometry

    _geometry(case, mo, path)
    expected = {
        tag: sorted(getattr(context, group))
        for tag, group in (("2", "active"), ("i", "inactive"), ("f", "frozen"), ("d", "deleted"))
    }
    if any(
        [i for i, value in enumerate(mo.type_indices) if value == tag] != indices
        for tag, indices in expected.items()
    ):
        raise ValueError("Projector context partition differs from the physical MO file")
    metadata = dict(
        format="openmolcas_NOSYM",
        ids=ids.tolist(),
        centers=list(mo.center_labels),
        coordinates=coordinates.tolist(),
        case={key: case.get(key) for key in ("basis", "charge", "relativistic", "group")},
    )
    for key in ("basis_by_element", "cholesky"):
        if key in case:
            metadata["case"][key] = copy.deepcopy(case[key])
    return mo.mo_coefficients, mo.ao_overlap, metadata


def _load_pyscf(context, path):
    molecule = context.payload.get("molecule")
    if not isinstance(molecule, str) or not molecule:
        raise ValueError("PySCF projector frame requires serialized molecule AO identities")
    from pyscf import gto

    mol = gto.loads(molecule)
    if identity(mol.atom) != context.geometry_id:
        raise ValueError("Projector frame geometry identity differs from its molecule")
    with np.load(path, allow_pickle=False) as data:
        coefficients = _real(data["coefficients"], "MO coefficients")
        overlap = _real(data["overlap"], "AO metric")
    calculated = mol.intor_symmetric("int1e_ovlp")
    if calculated.shape != overlap.shape or not np.allclose(
        calculated, overlap, atol=1e-10, rtol=0
    ):
        raise ValueError("PySCF NPZ AO metric differs from the serialized molecule")
    labels = mol.ao_labels(fmt=False)
    if len(labels) != context.nmo or len({tuple(row) for row in labels}) != context.nmo:
        raise ValueError("PySCF AO identities are missing or ambiguous")
    if mol.nelectron != context.total_electrons:
        raise ValueError("PySCF molecule electron count differs from its context")
    metadata = dict(
        format="pyscf",
        labels=[list(row) for row in labels],
        cart=bool(mol.cart),
        atom=mol._atm.tolist(),
        basis=mol._bas.tolist(),
        environment=mol._env.tolist(),
        charge=mol.charge,
        spin=mol.spin,
    )
    return coefficients, overlap, metadata


class ProjectorFrame:
    """One hash-bound complete MO basis in the common symmetric S**(1/2) gauge."""

    def __init__(self, context):
        self.context = Context.from_dict(copy.deepcopy(context.to_dict()))
        context = self.context
        self.context_id = context.context_id
        if not context.protocol_id or not context.effective_config_id or not context.geometry_id:
            raise ValueError(
                "Projector frame requires frozen protocol, configuration and geometry identities"
            )
        if not context.electron_roles:
            raise ValueError("Projector frame requires authenticated reference electron roles")
        raw_path = context.payload.get("basis_file")
        if not isinstance(raw_path, (str, Path)) or not str(raw_path):
            raise ValueError("Projector frame requires a physical basis file")
        path = Path(raw_path)
        if file_hash(path) != context.basis_id:
            raise ValueError("Projector basis hash differs from context.basis_id")
        expected_formats = (
            ("pyscf",)
            if context.backend == "pyscf"
            else ("native", electron_roles.SINGLE_CONFIGURATION_ROHF)
        )
        proof = context.payload.get("electron_role_provenance", {})
        if proof.get("reference", {}).get("format", "native") not in expected_formats:
            raise ValueError("Projector electron-role provenance has a different basis format")
        electron_roles.verify_context(context)
        partition = context.partition(context.active)
        if partition["inactive"] != sorted(context.inactive):
            raise ValueError("Inactive partition is not the affine reference-role complement")
        if context.backend == "pyscf":
            c, s, self._ao_identity = _load_pyscf(context, path)
            self.composition_target = copy.deepcopy(context.payload.get("target_ao"))
        elif context.backend in ("cas", "dmrg"):
            c, s, self._ao_identity = _load_native(context, path)
            self.composition_target = copy.deepcopy(context.payload["case"].get("target"))
        else:
            raise ValueError("Unsupported physical backend for projector response")
        c, s = _real(c, "MO coefficients"), _real(s, "AO metric")
        n = context.nmo
        if c.shape != (n, n) or s.shape != (n, n):
            raise ValueError("Projector frame requires a complete square MO basis")
        if not np.allclose(s, s.T, atol=1e-10, rtol=0):
            raise ValueError("Projector AO metric is not symmetric")
        if not np.allclose(c.T @ s @ c, np.eye(n), atol=1e-7, rtol=0):
            raise ValueError("Projector MO basis is not metric orthonormal")
        values, vectors = np.linalg.eigh((s + s.T) / 2)
        if values.min() <= 0:
            raise ValueError("Projector AO metric must be positive definite")
        u = ((vectors * np.sqrt(values)) @ vectors.T) @ c
        if not np.allclose(u.T @ u, np.eye(n), atol=1e-7, rtol=0):
            raise ValueError("Projector symmetric AO-coordinate transform is unstable")
        self._s = s
        self._projectors = np.einsum("ai,bi->iab", u, u)
        self.k, self.j = context.k, len(context.inactive)
        self.eta = 0.5 if self.j else 1.0
        self._frozen = self._projector(context.frozen)
        self._deleted = self._projector(context.deleted)
        self._scope = dict(
            geometry=context.geometry_id,
            backend=context.backend,
            protocol=context.protocol_id,
            config=context.effective_config_id,
            states=context.states,
            nmo=n,
            k=self.k,
            j=self.j,
            electrons=context.electrons,
            total_electrons=context.total_electrons,
            frozen_rank=len(context.frozen),
            deleted_rank=len(context.deleted),
            eta=self.eta,
        )
        if file_hash(path) != context.basis_id:
            raise ValueError("Projector basis changed while loading its frame")

    def _projector(self, indices):
        return self._projectors[list(indices)].sum(axis=0)

    def feature(self, active):
        partition = self.context.partition(active)
        pa = self._projector(partition["active"])
        pi = self._projector(partition["inactive"])
        return np.r_[
            1.0,
            np.sqrt(self.eta / self.k) * pa.ravel(),
            (np.sqrt((1 - self.eta) / self.j) if self.j else 0.0) * pi.ravel(),
        ] / np.sqrt(2)

    def affine_features(self):
        n = self.context.nmo
        result = np.zeros((n + 1, 1 + 2 * n * n))
        result[0, 0] = 1 / np.sqrt(2)
        projectors = self._projectors.reshape(n, n * n)
        result[1:, 1 : 1 + n * n] = np.sqrt(self.eta / (2 * self.k)) * projectors
        if self.j:
            double = np.array(
                [
                    int(i in self.context.window and role == 2)
                    for i, role in enumerate(self.context.electron_roles)
                ]
            )
            scaled = np.sqrt((1 - self.eta) / (2 * self.j)) * double[:, None] * projectors
            result[0, 1 + n * n :] = scaled.sum(axis=0)
            result[1:, 1 + n * n :] = -scaled
        return result

    def assert_compatible(self, other, *, composition=False):
        if not isinstance(other, ProjectorFrame) or self._scope != other._scope:
            raise ValueError(
                "Incompatible projector response scope (geometry, protocol, configuration or states)"
            )
        if self._ao_identity != other._ao_identity:
            raise ValueError("Incompatible physical AO identities or Hamiltonian metadata")
        if composition and (
            not self.composition_target
            or not other.composition_target
            or self.composition_target != other.composition_target
        ):
            raise ValueError("Enabled composition requires matching known target definitions")
        if not np.allclose(self._s, other._s, atol=1e-10, rtol=0):
            raise ValueError("Incompatible physical AO metrics")
        for name in ("_frozen", "_deleted"):
            if not np.allclose(getattr(self, name), getattr(other, name), atol=1e-7, rtol=0):
                raise ValueError("Incompatible physical frozen/deleted subspaces")


def polynomial_draw(affine, history_features, std, rng):
    """Joint squared-inner-product GP draw restricted to a finite feature span.

    Returns H and historical values with f(x)=[1,x]@H@[1,x].  Zero affine
    rows are supported for history-only draws.  No AO fourth-order tensor is
    formed; the orthogonal-coordinate factor has at most nmo+1+n_history rows.
    """
    affine = _real(affine, "Affine projector features")
    if affine.ndim != 2:
        raise ValueError("Affine projector features must be a matrix")
    history = _real(history_features, "Historical projector features")
    if history.size == 0:
        history = np.empty((0, affine.shape[1]))
    if history.ndim != 2 or history.shape[1] != affine.shape[1]:
        raise ValueError("Historical projector feature dimensions differ")
    if not np.isfinite(std) or std < 0:
        raise ValueError("Prior standard deviation must be finite and nonnegative")
    features = np.vstack((affine, history))
    if not len(features):
        return np.empty((0, 0)), np.empty(0)
    # Factor the features directly: taking a square root of their Gram matrix
    # amplifies roundoff-positive null eigenvalues and spoils shared-point draws.
    coordinates = np.linalg.qr(features.T, mode="r").T
    span_size = coordinates.shape[1]
    random = rng.normal(size=(span_size, span_size))
    random = std * (random + random.T) / 2
    current, old = coordinates[: len(affine)], coordinates[len(affine) :]
    matrix = current @ random @ current.T
    values = np.einsum("ir,rs,is->i", old, random, old)
    return (matrix + matrix.T) / 2, values
