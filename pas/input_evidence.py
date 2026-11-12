"""Authenticated historical input statistics, separate from physical receipts."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .common import file_hash, identity, read_json
from .contracts import Context


class MeasuredDomain:
    """Restrict proposals without changing the complete physical partition."""

    def __init__(self, context, window):
        self.context = context
        self.window = tuple(window)
        if (
            len(set(self.window)) != len(self.window)
            or any(type(i) is not int for i in self.window)
            or not set(self.window) <= set(context.window)
            or not set(context.active) <= set(self.window)
        ):
            raise ValueError("Diagnostic window must contain the parent active space")

    def __getattr__(self, name):
        if "context" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["context"], name)

    def partition(self, active):
        result = self.context.partition(active)
        if not set(result["active"]) <= set(self.window):
            raise ValueError("Candidate contains unmeasured orbitals")
        return result

    def to_dict(self):
        return self.context.to_dict()


def verify_binding(binding):
    if file_hash(binding["path"]) != binding["sha256"]:
        raise ValueError("Changed input-evidence source: " + binding["path"])


def cross_bind(document, raw):
    """Separate valid file hashes do not establish a shared physical calculation."""
    source = read_json(document["source_bindings"]["input_manifest"]["path"])
    evidence = read_json(document["source_bindings"]["diagnostic_evidence"]["path"])
    seed = read_json(document["initial_metric_sources"]["seed"]["path"])
    if (
        raw.get("status") != "available"
        or raw.get("entropy_unit") != "natural_log"
        or raw.get("mutual_information_convention") != "0.5*(s_i+s_j-s_ij)"
    ):
        raise ValueError("Diagnostic availability, entropy units or half-MI convention differs")
    protocols = [copy.deepcopy(row["diagnostic_protocol"]) for row in (source, document)]
    for protocol in protocols:
        protocol["source_evidence"].pop("path")
    for key in (
        "basis_id",
        "geometry_id",
        "states",
        "active",
        "inactive",
        "electrons",
        "total_electrons",
        "electron_roles",
        "composition",
    ):
        if document["context"].get(key) != source["context"].get(key):
            raise ValueError("Replay context differs from its source manifest: " + key)
    if (
        source["diagnostic"]["sha256"] != document["diagnostic"]["sha256"]
        or source["diagnostic_electrons"] != document["diagnostic_electrons"]
        or protocols[0] != protocols[1]
        or source["diagnostic_protocol"]["source_evidence"]["sha256"]
        != document["source_bindings"]["diagnostic_evidence"]["sha256"]
    ):
        raise ValueError("Diagnostic source and protocol do not share a bound manifest")
    if (
        not evidence["valid"]
        or evidence["fidelity"] != document["diagnostic_protocol"]["fidelity"]
        or evidence["observation"]["source_hashes"]["result"] != document["diagnostic"]["sha256"]
        or evidence["observation"]["wavefunction_id"]
        != raw["checkpoint"]["identity"]["aggregate_sha256"]
        or raw["checkpoint"]["fcidump_header"]["nelec"] != document["diagnostic_electrons"]
        or raw["checkpoint"]["fcidump_header"]["norb"] != len(raw["active_orbital_indices"])
    ):
        raise ValueError("Diagnostic wavefunction, fidelity or electron scope differs")
    if (
        seed["context"]["basis_id"] != source["context"]["basis_id"]
        or seed["context"]["active"] != source["context"]["active"]
        or seed["incumbent"]["artifacts"]["densities"]["sha256"]
        != document["initial_metric_sources"]["densities"]["sha256"]
    ):
        raise ValueError("Initial density does not belong to the bound physical parent")
    if document["initial_metrics"].get("qit") is not None:
        if seed["incumbent"]["metrics"].get("qit") != document["initial_metrics"]["qit"]:
            raise ValueError("An initial QIT value requires its own real parent evidence")


def load_historical_input(document):
    if document.get("schema") != "pas_offline_survey_replay_input_v1":
        raise ValueError("Expected an explicitly bound historical input manifest")
    for source in [
        document["diagnostic"],
        *document["initial_metric_sources"].values(),
        *document["source_bindings"].values(),
    ]:
        verify_binding(source)
    raw = read_json(document["diagnostic"]["path"])
    cross_bind(document, raw)
    return raw


def diagnostic_arrays(raw, context):
    if (
        raw["state_ids"] != context.states["ids"]
        or raw["state_weights"] != context.states["weights"]
    ):
        raise ValueError("Diagnostic states differ from the current physical context")
    rows = {row["state_id"]: row for row in raw["states"]}
    if len(rows) != len(raw["states"]) or set(rows) != set(context.states["ids"]):
        raise ValueError("Diagnostic has missing or duplicate states")
    entropy = np.array([rows[s]["single_orbital_entropy"] for s in context.states["ids"]])
    mi = np.array([rows[s]["mutual_information"] for s in context.states["ids"]])
    diagonal = sorted(
        [
            (order["bra_root"], matrix)
            for order, matrix in zip(
                raw["state_operator_order"], raw["state_operators"], strict=True
            )
            if order["bra_root"] == order["ket_root"]
        ],
        key=lambda row: row[0],
    )
    if [root for root, _ in diagonal] != list(range(len(context.states["ids"]))):
        raise ValueError("Missing or duplicate state density")
    return np.array([matrix for _, matrix in diagonal]), entropy, mi


def bootstrap_input(path, context, config):
    """Recondition original raw evidence; never import an output-response label."""
    from .backend import survey_quadratic

    if config.get("survey_model") != "posterior_evidence_capture_v1":
        raise ValueError("Bootstrap input evidence requires posterior capture")
    path = Path(path).resolve()
    source = dict(path=str(path), sha256=file_hash(path))
    document = read_json(path)
    raw = load_historical_input(document)
    original = Context.from_dict(document["context"])
    for key in (
        "basis_id",
        "geometry_id",
        "nmo",
        "active",
        "inactive",
        "frozen",
        "deleted",
        "electrons",
        "total_electrons",
        "states",
        "electron_roles",
    ):
        if identity(getattr(original, key)) != identity(getattr(context, key)):
            raise ValueError("Bootstrap input differs from the initial physical context: " + key)
    if (
        original.composition is None
        or context.composition is None
        or not np.allclose(original.composition, context.composition, atol=1e-12, rtol=0)
    ):
        raise ValueError("Bootstrap input differs from the initial physical AO projection")
    for ctx in (original, context):
        if (
            file_hash(ctx.payload["basis_file"]) != ctx.basis_id
            or file_hash(ctx.payload["checkpoint"]) != ctx.payload["checkpoint_hash"]
        ):
            raise ValueError("Bootstrap orbital files changed")
    if original.payload["checkpoint_hash"] != context.payload["checkpoint_hash"]:
        raise ValueError("Bootstrap checkpoint differs from the initial fixed-MO state")
    if config["dmrg_probe_m"] != document["diagnostic_protocol"]["fidelity"]:
        raise ValueError("Bootstrap fidelity differs from the declared input evidence model")
    window = raw["active_orbital_indices"]
    MeasuredDomain(context, window)
    density, entropy, mi = diagnostic_arrays(raw, context)
    partition = dict(active=window, electrons=document["diagnostic_electrons"])
    fields = survey_quadratic(context, partition, density, entropy, mi, config)
    verify_binding(source)
    return dict(
        schema="pas_bootstrap_input_evidence_v1",
        kind="bootstrap_input",
        valid=True,
        context_id=context.context_id,
        basis_id=context.basis_id,
        source=source,
        measured_window=list(window),
        physical_window=list(context.window),
        diagnostic=document["diagnostic"],
        original_protocol=document["diagnostic_protocol"],
        proposal_quadratic=fields,
        physical_receipt=False,
        target_response_label=False,
        validity_observation=False,
        new_physical_executions=0,
        new_physical_cpu_seconds=0.0,
        new_physical_wall_seconds=0.0,
        diagnostic_budget_charged=False,
        reuse_until_context_changes=True,
        event_index_offset=0,
        scope="historical_measured_input_not_current_protocol_receipt",
    )
