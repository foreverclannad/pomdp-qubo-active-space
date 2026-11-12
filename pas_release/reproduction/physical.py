"""Authenticate and reconstruct each physical endpoint in the five paper histories."""

import sys
from pathlib import Path

import numpy as np

from pas import acceptance, backend
from pas.common import identity
from pas.config import configuration
from pas.contracts import Context
from pas.diagnostic_window import derive_window, survey_partition
from pas.input_evidence import diagnostic_arrays

from .. import engine
from ..common import SOURCE, Evidence, compare_values, read_json, require, verify_source
from ..resources import DATA
from .raw_helpers import audit_target


def run(evidence_directory, group):
    source = verify_source()
    evidence = Evidence(evidence_directory)
    bindings = evidence.used
    groups = {"dio": ("NX1_DMRG", "CI_OO_DMRG"), "native": ("NX1_CAS", "CI_OO_CAS", "TiF3")}
    require(group in groups, "Physical replay group must be dio or native")
    cases = read_json(DATA / "cases.json")["cases"]
    report = dict(
        source_id=source,
        numerical_baseline_source_id=SOURCE,
        runs={},
        physical_jobs_launched=0,
        absolute_tolerance=1e-12,
    )
    with engine.read_only():
        for label in groups[group]:
            root = Path(cases[label]["original_run"])
            m, s, seed = [
                evidence.read(root / (n + ".json")) for n in ("manifest", "state", "seed")
            ]
            require(
                identity({k: v for k, v in m.items() if k != "manifest_id"}) == m["manifest_id"],
                "Physical run manifest content identity changed",
            )
            require(
                identity(seed) == m["seed_id"] and s["manifest_id"] == m["manifest_id"],
                "Physical run seed or state no longer belongs to its manifest",
            )
            cfg = configuration(m["config"])
            context = evidence.context(seed["context"])
            current = seed["incumbent"]
            first = audit_target(evidence, current, cfg)
            rows = [dict(event=0, kind="initial", **first)]
            limits = engine.budgets(m)
            nd, no = (limits["diagnostic_budget"], limits["optimization_budget"])
            for event in s["events"]:
                r = evidence.read(root / event["receipt"], event["sha256"])
                require(
                    r["evidence_id"] == event["evidence_id"],
                    "Event evidence identity differs from its signed receipt",
                )
                require(
                    r["context_id"] == context.context_id,
                    "Event receipt belongs to another orbital context",
                )
                require(
                    r["evidence_id"]
                    == identity({k: v for k, v in r.items() if k != "evidence_id"}),
                    "Physical receipt content identity changed",
                )
                for v in r["artifacts"].values():
                    if isinstance(v, dict) and {"path", "sha256"} <= v.keys():
                        evidence.artifact(v)
                if group != "dio":
                    evidence.request(
                        root / Path(event["receipt"]).parent,
                        context,
                        r,
                        cfg,
                        fixed_dmrg_policy=engine.fixed_policy(m, event["number"]),
                    )
                if r["kind"] == "optimize":
                    row = audit_target(evidence, r, cfg)
                    no -= 1
                else:
                    nd -= 1
                    row = {}
                    if group != "dio":
                        p = survey_partition(context, r["active"])
                        d = np.load(
                            evidence.artifact(r["artifacts"]["densities"]), allow_pickle=False
                        )
                        e = mi = None
                        if context.backend == "dmrg":
                            raw = read_json(evidence.artifact(r["artifacts"]["mps_evidence"]))
                            rd, e, mi = diagnostic_arrays(raw, context)
                            require(
                                np.max(abs(rd - d)) < 1e-12,
                                "MPS density reconstruction differs from the stored density by at least 1e-12",
                            )
                        fields = backend.survey_quadratic(context, p, d, e, mi, cfg)
                        survey_error = compare_values(
                            fields,
                            r["proposal_quadratic"],
                            location=f"{label}.{event['number']}.survey",
                        )
                        window = (
                            derive_window(context, m["survey_window_policy"])
                            if m.get("survey_window_policy")
                            else None
                        )
                        selected = window["active"] if window else list(context.window)
                        require(selected == r["active"], (label, event["number"], "window"))
                        row = dict(
                            raw_survey_reconstruction_passed=True,
                            survey_max_absolute_error=survey_error,
                            survey_window_reproduced=True,
                        )
                accepted = bool(
                    r["kind"] == "optimize"
                    and r["valid"]
                    and (
                        r["metrics"]["quality"]
                        > current["metrics"]["quality"] + cfg["acceptance_tolerance"]
                    )
                )
                require(
                    accepted == event["accepted"],
                    "Physical replay changed a recorded acceptance decision",
                )
                if accepted:
                    context = (
                        evidence.optimized(context, r)
                        if group != "dio"
                        else evidence.context(r["output_context"])
                    )
                    current = r
                row.update(event=event["number"], kind=r["kind"], accepted=accepted)
                rows.append(row)
            require(
                identity(current) == identity(s["incumbent"]),
                "Replayed incumbent differs from the recorded final incumbent",
            )
            require(
                context.context_id == Context.from_dict(s["context"]).context_id,
                "Replayed final orbital context differs from the recorded state",
            )
            require(
                (nd, no) == (s["remaining_diagnostics"], s["remaining_optimizations"]),
                "Replayed diagnostic or optimization budget differs from the recorded state",
            )
            checks = {}
            acc = s.get("acceptance")
            if acc:
                for name, comp in acc["comparisons"].items():
                    if comp["comparator_id"] == acceptance.RASORB_COMPARATOR_ID:
                        func = acceptance.compare_hdf5_to_rasorb
                    elif comp["comparator_id"] == acceptance.HDF5_COMPARATOR_ID:
                        func = acceptance.compare_hdf5_active_subspaces
                    else:
                        continue
                    reference = evidence.artifact(comp["reference"])
                    value = func(
                        evidence.bind(context.payload["basis_file"], context.basis_id),
                        reference,
                        expected_active_orbitals=context.k,
                    )["subspace"]
                    err = float(
                        np.max(
                            abs(
                                np.asarray(value["singular_values"])
                                - comp["subspace"]["singular_values"]
                            )
                        )
                    )
                    require(err < 1e-12, "Reference singular values changed by at least 1e-12")
                    require(
                        value["directions_at_or_above_0_95"]
                        == comp["subspace"]["directions_at_or_above_0_95"],
                        "Reference coverage differs from the recorded accepted directions",
                    )
                    checks[name] = dict(
                        directions=value["directions_at_or_above_0_95"],
                        dimension=value["dimension"],
                        minimum_singular_value=value["minimum_singular_value"],
                        singular_value_max_error=err,
                    )
            report["runs"][label] = dict(
                rows=rows,
                final=audit_target(evidence, current, cfg),
                reference_checks=checks,
                original_source_id=m["source_id"],
                config=cfg,
                status=s["status"],
                pending=s["pending"],
                final_state_reconstructed=True,
                remaining_diagnostics=nd,
                remaining_optimizations=no,
            )
            print("Raw evidence passed: " + label, file=sys.stderr, flush=True)
    evidence.unchanged()
    verify_source()
    report.update(files_unchanged=True, verified_file_count=len(bindings), file_bindings=bindings)
    return report
