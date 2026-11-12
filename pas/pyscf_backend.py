"""Small-system physical regression backend (PySCF's own optimizer).

This verifies the shared loop, not the production Molcas QN implementation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import metrics
from .common import file_hash, identity, read_json, write_json
from .config import configuration
from .contracts import Context


def _projection(c, overlap, target):
    block = overlap[np.ix_(target, target)]
    mixed = c.T @ overlap[:, target]
    return np.einsum("ia,ab,ib->i", mixed, np.linalg.inv(block), mixed).tolist()


def _solve(mol, c, ncas, electrons, weights, kind, config):
    from pyscf import fci, mcscf, scf

    mf = scf.RHF(mol)
    mc = (mcscf.CASSCF if kind == "optimize" else mcscf.CASCI)(mf, ncas, electrons)
    mc.fcisolver = fci.direct_spin0.FCI(mol)
    mc = mc.state_average_(weights)
    mc.fcisolver.conv_tol = 1e-10
    if kind == "optimize":
        mc.max_cycle_macro = config["max_macro"]
        mc.conv_tol = config["thr_energy"]
        mc.conv_tol_grad = config["thr_orbital"]
    mc.kernel(c)
    cis = mc.ci if isinstance(mc.ci, (tuple, list)) else [mc.ci]
    solver = fci.direct_spin0.FCI(mol)
    density = np.array([solver.make_rdm1(ci, ncas, (electrons // 2, electrons // 2)) for ci in cis])
    return mc, density


def initialize(directory, config):
    config = configuration(config)
    from pyscf import gto, lib, scf

    lib.num_threads(min(config["threads"], 2))
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mol = gto.M(atom="H 0 0 0; H 0 0 0.9; H 0 0 1.8; H 0 0 2.7", basis="sto-3g", spin=0, verbose=0)
    start = time.perf_counter()
    cpu = time.process_time()
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    if not mf.converged:
        raise RuntimeError("Small-system SCF failed")
    scf_basis = directory / "scf.npz"
    np.savez(
        scf_basis,
        coefficients=mf.mo_coeff,
        overlap=mf.get_ovlp(),
        occupations=mf.mo_occ,
        source_kind="pyscf_converged_scf",
        converged=True,
    )
    from . import electron_roles

    role_reference = dict(
        format="pyscf", basis_file=electron_roles.binding(scf_basis), active=[1, 2]
    )
    mc, d = _solve(mol, mf.mo_coeff, 2, 2, [0.5, 0.5], "optimize", config)
    if not mc.converged:
        raise RuntimeError("Small-system initial CASSCF failed")
    context, artifacts = _save(
        directory, mol, mc.mo_coeff, d, (1, 2), (0,), config, role_reference=role_reference
    )
    artifacts["electron_role_scf"] = electron_roles.binding(scf_basis)
    value = metrics.evaluate(
        config["channels"],
        config["weights"],
        densities=d,
        state_weights=[0.5, 0.5],
        electrons=2,
        composition=context.composition,
        active=context.active,
        composition_pair=context.composition_pair,
        composition_channels=context.composition_channels,
        composition_model=config["composition_model"],
        composition_preferences=context.composition_preferences,
        composition_purity_weight=config.get("composition_purity_weight", 0.5),
    )
    value["energy"] = float(mc.e_tot)
    init = dict(
        role="physical_small_system_regression",
        optimizer="PySCF native",
        fresh_scf=True,
        solver_executions=2,
        cpu_seconds=time.process_time() - cpu,
        wall_seconds=time.perf_counter() - start,
        matches_search_protocol=True,
    )
    receipt = dict(
        kind="initial",
        active=list(context.active),
        context_id=context.context_id,
        request_id=identity(init),
        valid=True,
        converged=True,
        state_valid=True,
        metrics=value,
        artifacts=artifacts,
        output_context=context.to_dict(),
        **{k: init[k] for k in ("cpu_seconds", "wall_seconds")},
    )
    receipt["evidence_id"] = identity(receipt)
    seed = dict(context=context.to_dict(), incumbent=receipt, initialization=init)
    write_json(directory / "seed.json", seed)
    return seed


def _save(
    directory,
    mol,
    c,
    d,
    active,
    inactive,
    config,
    *,
    role_reference=None,
    reference_context=None,
    input_partition=None,
):
    basis = directory / "orbitals.npz"
    overlap = mol.intor_symmetric("int1e_ovlp")
    np.savez(basis, coefficients=c, overlap=overlap, densities=d)
    target = [0, 1]
    pair = None
    channels = None
    audit = None
    preferences = None
    if "composition" in config["channels"] and (
        config["composition_model"] in metrics.COMPLEMENTARITY_COMPOSITION_MODELS
        or config.get("composition_preferences") is not None
    ):
        from types import SimpleNamespace

        from .composition import determinant_pair_matrix, harmonic_channel_data

        # This regression molecule has one 1s AO per hydrogen; the target is
        # exactly the same first two centres used by the projection score.
        labels = mol.ao_labels(fmt=False)
        if any(label[2] != "1s" for label in labels) or len(labels) != mol.natm:
            raise ValueError("Pair regression adapter requires one 1s AO per centre")
        orbitals = SimpleNamespace(
            mo_coefficients=c,
            ao_overlap=overlap,
            orbital_count=c.shape[1],
            basis_function_ids=np.array([[label[0] + 1, 1, 0, 0] for label in labels]),
            center_labels=tuple(mol.atom_symbol(i) for i in range(mol.natm)),
        )
        if config["composition_model"] == "harmonic_channel_pair_v1":
            channels, audit = harmonic_channel_data(orbitals, [dict(center_indices=target)])
        elif config["composition_model"] in metrics.PAIR_COMPOSITION_MODELS:
            matrix, audit = determinant_pair_matrix(orbitals, [dict(center_indices=target)])
            pair = matrix.tolist()
        from .preferences import build_preferences

        preferences = build_preferences(orbitals, config.get("composition_preferences"))
    context = Context(
        basis_id=file_hash(basis),
        geometry_id=identity(mol.atom),
        backend="pyscf",
        nmo=c.shape[1],
        active=tuple(active),
        inactive=tuple(inactive),
        total_electrons=mol.nelectron,
        electrons=2,
        composition=_projection(c, overlap, target)
        if "composition" in config["channels"]
        else None,
        composition_pair=pair,
        composition_channels=channels,
        composition_preferences=preferences,
        effective_config_id=identity(config),
        protocol_id=identity(dict(backend="pyscf", config=config)),
        payload=dict(molecule=mol.dumps(), basis_file=str(basis), target_ao=target),
    )
    if audit is not None:
        context.payload["composition_audit"] = audit
    from . import electron_roles

    if role_reference is not None:
        roles, proof = electron_roles.from_reference(
            role_reference, electron_roles.binding(basis), electron_roles.partition(context)
        )
    elif reference_context is not None:
        roles, proof = electron_roles.from_context(
            reference_context,
            input_partition,
            electron_roles.binding(basis),
            electron_roles.partition(context),
        )
    else:
        raise ValueError("Small-system context requires an SCF electron-role source")
    context.electron_roles = tuple(roles)
    context.payload["electron_role_provenance"] = proof
    context.__post_init__()
    return context, dict(basis_file=dict(path=str(basis), sha256=file_hash(basis)))


def execute(context, active, kind, directory, config):
    from pyscf import gto, lib

    config = configuration(config)
    from .preferences import assert_preference_binding

    assert_preference_binding(context, config)
    lib.num_threads(min(config["threads"], 2))
    if "qit" in config["channels"] or kind == "survey":
        raise ValueError(
            "The small-system verification adapter supports target occupation/composition only"
        )
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if file_hash(context.payload["basis_file"]) != context.basis_id:
        raise ValueError("Small-system input basis changed")
    from . import electron_roles

    electron_roles.verify_context(context)
    if kind not in ("probe", "optimize"):
        raise ValueError("Unsupported small-system physical action")
    part = context.partition(active)
    settings = dict(
        backend="pyscf",
        optimizer="PySCF native" if kind == "optimize" else None,
        fci_solver="direct_spin0",
        fci_tolerance=1e-10,
        states=context.states,
        max_macro=config["max_macro"],
        thr_energy=config["thr_energy"],
        thr_orbital=config["thr_orbital"],
    )
    request = dict(
        schema="pas_physical_request_v1",
        context_id=context.context_id,
        basis_id=context.basis_id,
        checkpoint_hash=context.payload.get("checkpoint_hash", context.basis_id),
        kind=kind,
        partition=part,
        protocol=config,
        input=json.dumps(settings, sort_keys=True, separators=(",", ":")),
    )
    request["request_id"] = identity(request)
    if (directory / "receipt.json").exists():
        from .controller import _verify_receipt, _verify_request

        receipt = read_json(directory / "receipt.json")
        _verify_receipt(receipt)
        _verify_request(directory, context, receipt, config)
        if receipt["request_id"] != request["request_id"]:
            raise ValueError("Cached PySCF request differs")
        return receipt
    if (directory / "started.json").exists():
        raise RuntimeError("Interrupted action cannot be automatically rerun")
    if (directory / "request.json").exists() and read_json(directory / "request.json") != request:
        raise ValueError("Existing PySCF request differs")
    order = part["inactive"] + part["active"] + part["secondary"]
    c = np.load(context.payload["basis_file"])["coefficients"][:, order]
    mol = gto.loads(context.payload["molecule"])
    write_json(directory / "request.json", request)
    write_json(directory / "started.json", dict(time_unix=time.time()))
    start = time.perf_counter()
    cpu = time.process_time()
    mc, d = _solve(mol, c, context.k, context.electrons, context.states["weights"], kind, config)
    receipt = dict(
        request_id=request["request_id"],
        context_id=context.context_id,
        kind=kind,
        active=sorted(active),
        valid=bool(mc.converged),
        converged=bool(mc.converged),
        state_valid=bool(mc.converged),
        metrics={},
        cpu_seconds=time.process_time() - cpu,
        wall_seconds=time.perf_counter() - start,
        provenance=dict(optimizer="PySCF native"),
        artifacts=dict(
            request=dict(
                path=str(directory / "request.json"), sha256=file_hash(directory / "request.json")
            )
        ),
    )
    if mc.converged:
        ni = len(part["inactive"])
        output, artifacts = _save(
            directory,
            mol,
            mc.mo_coeff,
            d,
            tuple(range(ni, ni + context.k)),
            tuple(range(ni)),
            config,
            reference_context=context,
            input_partition=part,
        )
        receipt.update(output_context=output.to_dict())
        receipt["artifacts"].update(artifacts)
        receipt["metrics"] = metrics.evaluate(
            config["channels"],
            config["weights"],
            densities=d,
            state_weights=context.states["weights"],
            electrons=context.electrons,
            composition=output.composition,
            active=output.active,
            composition_pair=output.composition_pair,
            composition_channels=output.composition_channels,
            composition_model=config["composition_model"],
            composition_preferences=output.composition_preferences,
            composition_purity_weight=config.get("composition_purity_weight", 0.5),
        )
        receipt["metrics"]["energy"] = float(mc.e_tot)
    else:
        receipt["failure"] = "numerical_nonconvergence"
    receipt["evidence_id"] = identity(receipt)
    write_json(directory / "receipt.json", receipt)
    return receipt
