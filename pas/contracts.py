"""Small shared contracts; arrays live in physical artifacts, not policy state."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .common import identity


@dataclass
class Context:
    basis_id: str
    geometry_id: str
    backend: str
    nmo: int
    active: tuple[int, ...]
    inactive: tuple[int, ...]
    total_electrons: int
    electrons: int = 16
    frozen: tuple[int, ...] = ()
    deleted: tuple[int, ...] = ()
    stable_order: tuple[int, ...] = ()
    electron_roles: tuple[int, ...] = ()
    composition: list[float] | None = None
    composition_pair: list[list[float]] | None = None
    composition_channels: dict | None = None
    states: dict = field(
        default_factory=lambda: dict(roots=[1, 2], weights=[0.5, 0.5], spin_multiplicity=1)
    )
    protocol_id: str = ""
    effective_config_id: str = ""
    payload: dict = field(default_factory=dict)
    required_active: tuple[int, ...] = ()
    composition_preferences: dict | None = None

    def __post_init__(self):
        if any(type(v) is not int for v in (self.nmo, self.total_electrons, self.electrons)):
            raise ValueError("Orbital/electron counts must be integers")
        for key in ("active", "inactive", "frozen", "deleted", "stable_order", "required_active"):
            setattr(self, key, tuple(getattr(self, key)))
            values = getattr(self, key)
            if any(type(i) is not int for i in values) or len(set(values)) != len(values):
                raise ValueError("Orbital indices must be distinct integers")
        if not self.stable_order:
            self.stable_order = tuple(range(self.nmo))
        if sorted(self.stable_order) != list(range(self.nmo)):
            raise ValueError("stable_order is not a permutation")
        groups = [set(getattr(self, k)) for k in ("active", "inactive", "frozen", "deleted")]
        if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
            raise ValueError("Orbital partitions overlap")
        if any(i < 0 or i >= self.nmo for group in groups for i in group):
            raise ValueError("Orbital index outside basis")
        if not set(self.required_active) <= set(self.active):
            raise ValueError("Required active orbitals must belong to the active space")
        if self.total_electrons != self.electrons + 2 * (len(self.inactive) + len(self.frozen)):
            raise ValueError("Electron partition mismatch")
        if not self.active or not 0 <= self.electrons <= 2 * len(self.active):
            raise ValueError("Invalid active electron/orbital count")
        self.electron_roles = tuple(self.electron_roles)
        if self.electron_roles:
            roles = self.electron_roles
            if len(roles) != self.nmo or any(
                type(r) is not int or r not in (0, 1, 2) for r in roles
            ):
                raise ValueError("Electron roles require one integer 0/1/2 per orbital")
            if (
                sum(roles) != self.total_electrons
                or sum(roles[i] for i in self.active) != self.electrons
            ):
                raise ValueError(
                    "Reference electron roles differ from total or active electron count"
                )
            if any(roles[i] != 2 for i in self.inactive + self.frozen):
                raise ValueError("Inactive and frozen orbitals require reference role 2")
            occupied_partition = set(self.active + self.inactive + self.frozen)
            if any(roles[i] != 0 for i in range(self.nmo) if i not in occupied_partition):
                raise ValueError(
                    "Secondary and deleted orbitals require reference role 0; every single must be active"
                )
        roots, weights = self.states.get("roots", []), self.states.get("weights", [])
        if (
            not roots
            or len(roots) != len(weights)
            or len(set(roots)) != len(roots)
            or any(type(r) is not int or r < 1 for r in roots)
            or any(not math.isfinite(w) or w <= 0 for w in weights)
            or not math.isclose(sum(weights), 1.0, abs_tol=1e-12, rel_tol=0)
        ):
            raise ValueError("State roots and normalized positive weights are required")
        spin = self.states.get("spin_multiplicity", 1) - 1
        if (
            type(spin) is not int
            or spin < 0
            or spin > min(self.electrons, 2 * len(self.active) - self.electrons)
            or (self.electrons - spin) % 2
        ):
            raise ValueError("Active electron/spin parity or spin bound is invalid")
        if self.composition is not None and len(self.composition) != self.nmo:
            raise ValueError("Composition must cover the complete input MO basis")
        if self.composition is not None and any(
            not math.isfinite(v) or not -1e-7 <= v <= 1 + 1e-7 for v in self.composition
        ):
            raise ValueError("Composition projections must be finite physical values")
        if self.composition_pair is not None:
            from .metrics import validate_composition_pair

            self.composition_pair = validate_composition_pair(
                self.composition_pair, self.nmo
            ).tolist()
        if self.composition_channels is not None:
            from .metrics import validate_composition_channels

            self.composition_channels = validate_composition_channels(
                self.composition_channels, self.nmo
            )
        if self.composition_preferences is not None:
            from .preferences import validate_preferences

            if self.composition is None:
                raise ValueError("Composition preferences require base AO projections")
            self.composition_preferences = validate_preferences(
                self.composition_preferences, self.nmo
            )

    @property
    def k(self):
        return len(self.active)

    @property
    def window(self):
        return tuple(i for i in range(self.nmo) if i not in set(self.frozen + self.deleted))

    @property
    def context_id(self):
        fields = self.to_dict()
        fields.pop("payload")
        return identity(fields)

    def partition(self, active):
        active = tuple(sorted(active))
        if (
            any(type(i) is not int for i in active)
            or len(active) != self.k
            or len(set(active)) != self.k
            or not set(active) <= set(self.window)
        ):
            raise ValueError("Candidate must contain exactly K distinct eligible orbitals")
        if not set(self.required_active) <= set(active):
            raise ValueError("Candidate must retain every required active orbital")
        ni = (self.total_electrons - self.electrons) // 2 - len(self.frozen)
        if self.electron_roles:
            roles = self.electron_roles
            if sum(roles[i] for i in active) != self.electrons:
                raise ValueError("Candidate violates the hard reference-electron budget")
            if any(role == 1 and i not in active for i, role in enumerate(roles)):
                raise ValueError("Every reference singly occupied orbital must remain active")
            inactive = [i for i in self.window if i not in active and roles[i] == 2]
            if len(inactive) != ni:
                raise ValueError("Reference roles leave an invalid inactive complement")
            return dict(
                active=list(active),
                inactive=inactive,
                frozen=list(self.frozen),
                secondary=[i for i in self.window if i not in active and roles[i] == 0],
                deleted=list(self.deleted),
                electrons=self.electrons,
            )
        order = [i for i in self.stable_order if i in self.inactive]
        order += [i for i in self.stable_order if i not in self.inactive]
        eligible = [i for i in order if i in self.window and i not in active]
        if len(eligible) < ni:
            raise ValueError("Insufficient inactive complement")
        return dict(
            active=list(active),
            inactive=sorted(eligible[:ni]),
            frozen=list(self.frozen),
            secondary=sorted(set(eligible[ni:])),
            deleted=list(self.deleted),
            electrons=self.electrons,
        )

    def to_dict(self):
        result = asdict(self)
        # Preserve the serialized identity of projection-only legacy contexts.
        if self.composition_pair is None:
            result.pop("composition_pair")
        if self.composition_channels is None:
            result.pop("composition_channels")
        if not self.required_active:
            result.pop("required_active")
        if self.composition_preferences is None:
            result.pop("composition_preferences")
        return result

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
