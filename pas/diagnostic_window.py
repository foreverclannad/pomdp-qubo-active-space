"""Bounded diagnostic scopes from the historical AO/frontier threshold rule."""

from __future__ import annotations

import copy

import h5py
import numpy as np

from .common import file_hash, identity
from .contracts import Context

HISTORICAL_WINDOW_POLICY = dict(
    schema="pas_diagnostic_window_policy_v1",
    model="historical_ao_frontier_union_v1",
    minimum_target_projection=0.05,
    minimum_frontier_score=0.05,
    frontier_temperature_hartree=0.25,
)


def validate_policy(policy):
    """Keep the historical recipe, with an optional declared resource ceiling."""
    if (
        not isinstance(policy, dict)
        or {key: value for key, value in policy.items() if key != "max_orbitals"}
        != HISTORICAL_WINDOW_POLICY
    ):
        raise ValueError("Unsupported diagnostic window policy")
    if "max_orbitals" in policy and (
        type(policy["max_orbitals"]) is not int or policy["max_orbitals"] < 1
    ):
        raise ValueError("Diagnostic window policy max_orbitals must be a positive integer")
    return copy.deepcopy(policy)


def survey_partition(context, active):
    """Rebuild a complete legal partition for an explicitly supplied survey scope.

    This validates old requests by their actual scope, without applying a newer
    window policy. Full-window requests also retain the legacy no-role behavior.
    """
    context = Context.from_dict(context.to_dict())
    active = list(active)
    if (
        any(type(i) is not int for i in active)
        or len(set(active)) != len(active)
        or not set(active) <= set(context.window)
        or not set(context.active) <= set(active)
    ):
        raise ValueError(
            "Survey window must contain the current active space and distinct eligible orbitals"
        )
    active = sorted(active)
    outside = [i for i in context.window if i not in active]
    if not context.electron_roles:
        if outside:
            raise ValueError("A bounded survey window requires reference electron roles")
        inactive, secondary = [], []
    else:
        roles = context.electron_roles
        if any(roles[i] == 1 for i in outside):
            raise ValueError("Every singly occupied reference orbital must remain in the survey")
        inactive = [i for i in outside if roles[i] == 2]
        secondary = [i for i in outside if roles[i] == 0]
    electrons = context.total_electrons - 2 * (len(inactive) + len(context.frozen))
    if context.electron_roles and sum(context.electron_roles[i] for i in active) != electrons:
        raise ValueError("Survey electron roles differ from its complete physical partition")
    # Reuse the physical contract for electron bounds, spin parity, and partitions.
    fields = context.to_dict()
    fields.update(active=active, inactive=inactive, electrons=electrons)
    Context.from_dict(fields)
    return dict(
        active=active,
        inactive=inactive,
        secondary=secondary,
        frozen=list(context.frozen),
        deleted=list(context.deleted),
        electrons=electrons,
    )


def derive_window(context, policy=None):
    """Read the bound input H5; select thresholds without a solver or posterior."""
    policy = validate_policy(HISTORICAL_WINDOW_POLICY if policy is None else policy)
    context = Context.from_dict(context.to_dict())
    maximum = policy.get("max_orbitals")
    if maximum is not None and maximum < context.k:
        raise ValueError(
            "Diagnostic window max_orbitals cannot be smaller than the current active space"
        )
    if not context.electron_roles or context.composition is None:
        raise ValueError(
            "Diagnostic window policy requires AO composition and reference electron roles"
        )
    path = context.payload.get("basis_file")
    if not path or file_hash(path) != context.basis_id:
        raise ValueError("Diagnostic window input basis hash differs")
    with h5py.File(path, "r") as handle:
        if "MO_ENERGIES" not in handle:
            raise ValueError("Diagnostic window input H5 lacks MO_ENERGIES")
        energies = np.asarray(handle["MO_ENERGIES"][()], dtype=float)
    if energies.shape != (context.nmo,) or not np.all(np.isfinite(energies)):
        raise ValueError("Diagnostic window requires one finite MO energy per orbital")
    if file_hash(path) != context.basis_id:
        raise ValueError("Diagnostic window input basis changed while reading")
    roles = np.asarray(context.electron_roles)
    occupied, empty = roles > 1e-8, roles <= 1e-8
    # Match the original vendor implementation, including its one-class fallback.
    mu = (
        0.5 * (float(np.max(energies[occupied])) + float(np.min(energies[empty])))
        if np.any(occupied) and np.any(empty)
        else float(np.median(energies))
    )
    frontier = np.exp(-np.abs(energies - mu) / policy["frontier_temperature_hartree"])
    composition = np.asarray(context.composition, dtype=float)
    passed = [
        i
        for i in context.window
        if composition[i] >= policy["minimum_target_projection"]
        or frontier[i] >= policy["minimum_frontier_score"]
    ]
    active = sorted(set(passed) | set(context.active))
    cap_audit = None
    if maximum is not None:
        # The physical contract already places every reference single in active.
        # Retain that complete space before ranking optional threshold members.
        mandatory = set(context.active)
        order = {orbital: position for position, orbital in enumerate(context.stable_order)}
        projection_strength = composition / policy["minimum_target_projection"]
        frontier_strength = frontier / policy["minimum_frontier_score"]
        ranked = sorted(
            set(passed) - mandatory,
            key=lambda i: (
                -max(projection_strength[i], frontier_strength[i]),
                -min(projection_strength[i], frontier_strength[i]),
                order[i],
            ),
        )
        selected = ranked[: maximum - len(mandatory)]
        cap_audit = dict(
            max_orbitals=maximum,
            applied=len(active) > maximum,
            before_count=len(active),
            after_count=len(mandatory) + len(selected),
            ranking="descending_max_then_min_of_threshold_normalized_AO_frontier_then_stable_order",
            mandatory_active=sorted(mandatory),
            ranked_optional=ranked,
            selected_optional=selected,
            discarded_optional=ranked[len(selected) :],
        )
        active = sorted(mandatory | set(selected))
    partition = survey_partition(context, active)
    record = dict(
        schema="pas_diagnostic_window_v1",
        context_id=context.context_id,
        basis_id=context.basis_id,
        policy=policy,
        active=active,
        partition=partition,
        descriptor_audit=dict(
            energy_source="bound_input_H5_MO_ENERGIES",
            energies_hartree=energies.tolist(),
            chemical_potential_hartree=mu,
            target_projection=composition.tolist(),
            frontier=frontier.tolist(),
            eligible_window=list(context.window),
            threshold_passed=passed,
            incumbent_additions=sorted(set(context.active) - set(passed)),
            role_source="context_reference_electron_roles",
            natural_occupations_used=False,
            posterior_used=False,
        ),
    )
    if cap_audit is not None:
        record["descriptor_audit"]["orbital_cap"] = cap_audit
    return {**record, "window_id": identity(record)}
