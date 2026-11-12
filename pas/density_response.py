"""Input-density partition loss, never a terminal quality or a solver receipt."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from . import electron_roles, metrics
from .common import file_hash
from .contracts import Context


class DensityLoss:
    """Hash-bound reference D+V in electrons, affine in the current role domain.

    Normalization by target active electrons is a fixed modeling scale, not a
    physical error bound or a calibrated probability. Historical responses must
    retain the DensityLoss of their original input Context.
    """

    def __init__(self, context):
        self.context = Context.from_dict(copy.deepcopy(context.to_dict()))
        context = self.context
        if context.backend not in ("cas", "pyscf"):
            raise ValueError("Density response currently supports CAS and PySCF only")
        if not context.electron_roles or context.electrons <= 0:
            raise ValueError(
                "Density response requires electron roles and positive active electrons"
            )
        raw_path = context.payload.get("basis_file")
        if not isinstance(raw_path, (str, Path)) or not str(raw_path):
            raise ValueError("Density response requires a physical basis file")
        path = Path(raw_path)
        if file_hash(path) != context.basis_id:
            raise ValueError("Density response basis hash differs from context.basis_id")
        electron_roles.verify_context(context)
        current = context.partition(context.active)
        if current["inactive"] != sorted(context.inactive):
            raise ValueError("Density response requires the actual affine inactive complement")
        if context.backend == "cas":
            from .backend import read_densities
            from .orbital_io import read_openmolcas_orbital_hdf5

            densities, _ = read_densities(
                path, context.states, context.electrons, require_natural=False
            )
            mo = read_openmolcas_orbital_hdf5(path)
            expected = {
                tag: sorted(getattr(context, name))
                for tag, name in (
                    ("2", "active"),
                    ("i", "inactive"),
                    ("f", "frozen"),
                    ("d", "deleted"),
                )
            }
            if any(
                [i for i, tag in enumerate(mo.type_indices) if tag == label] != indices
                for label, indices in expected.items()
            ):
                raise ValueError("Density response MO partition differs from context")
        else:
            # The local adapter stores roots in context.states order in this
            # same hash-bound NPZ; no per-root natural-orbital reordering occurs.
            with np.load(path, allow_pickle=False) as data:
                densities = np.asarray(data["densities"])
        densities = metrics.validate_densities(
            densities, context.states["weights"], context.electrons
        )
        if densities.shape != (len(context.states["roots"]), context.k, context.k):
            raise ValueError("Density response root/active dimensions differ from context")
        averaged = np.einsum("r,rij->ij", context.states["weights"], densities)
        occupations = np.diag(averaged).real
        if context.backend == "cas" and not np.allclose(
            occupations, mo.occupations[list(context.active)], atol=1e-6, rtol=0
        ):
            raise ValueError("Density response diagonal differs from native MO occupations")
        self.occupations = np.zeros(context.nmo)
        self.occupations[list(context.inactive + context.frozen)] = 2.0
        self.occupations[list(context.active)] = occupations
        if not np.isclose(self.occupations.sum(), context.total_electrons, atol=1e-6, rtol=0):
            raise ValueError("Density response total electron trace differs from context")
        if context.backend == "cas" and not np.allclose(
            self.occupations, mo.occupations, atol=1e-6, rtol=0
        ):
            raise ValueError("Density response complete occupations differ from native partition")
        # Frozen/deleted orbitals are fixed external partitions, with zero loss.
        weights = np.zeros(context.nmo)
        for i in context.window:
            role = context.electron_roles[i]
            weights[i] = (
                2 - self.occupations[i] if role == 2 else self.occupations[i] if role == 0 else 0.0
            )
        self._affine = np.r_[weights.sum(), -weights]
        self.scale = float(context.electrons)
        if abs(self.values([context.active])[0]) > 1e-6:
            raise ValueError("Density response reference partition has nonzero loss")
        if file_hash(path) != context.basis_id:
            raise ValueError("Density response basis changed while reading density")

    def affine(self):
        """Return [constant, x_0 coefficient, ...] for raw L in electrons."""
        return self._affine.copy()

    def values(self, active_list):
        """Return raw L for legal active sets; divide by scale in the model."""
        result = []
        for active in active_list:
            active = self.context.partition(active)["active"]
            result.append(self._affine[0] + self._affine[np.asarray(active) + 1].sum())
        return np.asarray(result, dtype=float)

    def audit(self, active):
        parts = self.context.partition(active)
        d = float(np.sum(2 - self.occupations[parts["inactive"]]))
        v = float(np.sum(self.occupations[parts["secondary"]]))
        expected = float(np.sum(self.occupations[parts["active"]]))
        return dict(
            D=d,
            V=v,
            L=d + v,
            delta_n=expected - self.context.electrons,
            active_expected=expected,
            normalized=(d + v) / self.scale,
            normalization_electrons=self.context.electrons,
            basis_id=self.context.basis_id,
            context_id=self.context.context_id,
        )
