"""Replay the fixed-pool A/B/C ablations and the CI-CC subspace comparison."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from pas import acceptance, planner
from pas.config import configuration, planning_configuration

from .. import engine
from ..common import SOURCE, Evidence, require, verify_source
from .raw_helpers import audit_target


def run(evidence_directory):
    source = verify_source()
    evidence = Evidence(evidence_directory)
    bindings = evidence.used
    original = Path("/localdata/lichen/pas_cas_nc_20261014_r1/NX1/run")
    stage = Path("/home/lichen/staging/pas_nx1_round4_controls_20261015_r1")
    controls = Path("/localdata/lichen/pas_nx1_round4_controls_20261015_r1")
    m = evidence.read(original / "manifest.json")
    seed = evidence.read(original / "seed.json")
    state = evidence.read(original / "state.json")
    audit = evidence.read(
        stage / "decision_audit.json",
        "36169a73ab9d2d34fef43962403349ed3dfd8200ea4577ab5d5ee968d3069549",
    )
    cfg = planning_configuration(m["config"], m.get("response_policy"))
    context = evidence.context(seed["context"])
    seed = evidence.relocate(seed)
    receipts = []
    quality = seed["incumbent"]["metrics"]["quality"]
    budgets = engine.budgets(m)
    nd, no = (budgets["diagnostic_budget"], budgets["optimization_budget"])
    history = []
    fourth = None

    def choose_fixed(pool, generation, domain, belief, q, nd, no, args):
        with patch.object(
            planner, "propose", return_value=([tuple(x) for x in pool], deepcopy(generation))
        ):
            return planner.choose(domain, belief, q, nd, no, cfg, **args)

    with engine.read_only():
        for ev in state["events"]:
            r = evidence.read(original / ev["receipt"], ev["sha256"])
            if r["kind"] == "optimize":
                actual = evidence.read(original / "decisions" / f"{ev['number']:03d}.json")
                base = engine.belief(context, cfg, receipts, seed, None)
                domain, survey, _ = engine.input_domain(context, None, receipts)
                excluded = {
                    k: [
                        x["active"]
                        for x in receipts
                        if x["kind"] == k and x["context_id"] == context.context_id
                    ]
                    for k in ("probe", "optimize")
                }
                kwargs = dict(
                    event_index=len(receipts),
                    excluded=excluded,
                    survey=survey,
                    planner_policy=m["planner_policy"],
                )
                base.bind_input_evidence(domain, survey["proposal_quadratic"])
                empty = deepcopy(base)
                empty.observations = {k: [] for k in empty.observations}
                empty._cache = {}
                estimates = {}
                for name, b in [("full", base), ("no_response_updates", empty)]:
                    means = {
                        c: float(b.predict([r["active"]], "optimize", c)[0][0])
                        for c in cfg["channels"]
                    }
                    mean = sum((cfg["weights"][c] * means[c] for c in cfg["channels"])) / sum(
                        cfg["weights"].values()
                    )
                    estimates[name] = dict(
                        mean=mean, absolute_error=abs(mean - r["metrics"]["quality"])
                    )
                history.append(
                    dict(
                        event=ev["number"],
                        actual_quality=r["metrics"]["quality"],
                        predictions=estimates,
                    )
                )
                if len(history) == 4:
                    pool = actual["pool"]
                    generation = actual["audit"]["generation"]
                    choices = {}
                    choices["full"] = choose_fixed(
                        pool, generation, domain, deepcopy(base), quality, nd, no, kwargs
                    )
                    choices["no_response_updates"] = choose_fixed(
                        pool, generation, domain, empty, quality, nd, no, kwargs
                    )
                    original_values = planner.action_values

                    def diagonal_only(*a, **kw):
                        direct, high, weights, ph, pl = original_values(*a, **kw)
                        high = {c: (mu, np.diag(np.diag(cov))) for c, (mu, cov) in high.items()}
                        return (direct, high, weights, ph, pl)

                    with patch.object(planner, "action_values", side_effect=diagonal_only):
                        choices["no_h2_crosscov"] = choose_fixed(
                            pool, generation, domain, deepcopy(base), quality, nd, no, kwargs
                        )
                    archived = audit["rounds"][3]["variants"]
                    for name, d in choices.items():
                        require(
                            [i + 1 for i in d["active"]] == archived[name]["chosen_one_based"], name
                        )
                    require(
                        choices["full"]["active"] == actual["active"],
                        "Control A no longer reproduces the recorded orbital choice",
                    )
                    fourth = dict(
                        pool=pool,
                        choices={
                            name: dict(
                                active=d["active"],
                                active_one_based=[x + 1 for x in d["active"]],
                                values=d["audit"]["optimize_values"],
                            )
                            for name, d in choices.items()
                        },
                        incumbent_quality=quality,
                    )
            accepted = bool(
                r["kind"] == "optimize"
                and r["valid"]
                and (r["metrics"]["quality"] > quality + cfg["acceptance_tolerance"])
            )
            require(
                accepted == ev["accepted"],
                "Control replay changed the recorded acceptance decision",
            )
            if accepted:
                context = evidence.optimized(context, r)
                quality = r["metrics"]["quality"]
            receipts.append(evidence.relocate(r))
            if r["kind"] == "optimize":
                no -= 1
            else:
                nd -= 1
        require(
            fourth is not None, "Fourth optimization decision is missing from the control history"
        )
        endpoints = {}
        ref = Path(
            "/home/lichen/staging/pas_cas_nc_20261014_r1/inputs/references/45d66f7ac9b2d404-reference_nx1.RasOrb"
        )
        evidence.bind(ref, "45d66f7ac9b2d404319ddcf874acebe52ef86a98222ff0af677e23c55e8ea716")
        input_context = evidence.context(evidence.read(controls / "input_context.json"))
        for label, variant, path in [
            ("A", "full", original / "actions/008_optimize/receipt.json"),
            ("B", "no_h2_crosscov", controls / "B_no_h2_crosscov/action/receipt.json"),
            ("C", "no_response_updates", controls / "C_no_response_updates/action/receipt.json"),
        ]:
            r = evidence.read(path)
            require(
                r["active"] == fourth["choices"][variant]["active"],
                "Control endpoint orbitals differ from the selected candidate",
            )
            evidence.request(path.parent, input_context, r, configuration(m["config"]))
            row = audit_target(evidence, r, configuration(m["config"]))
            compare = acceptance.compare_hdf5_to_rasorb(
                evidence.artifact(r["artifacts"]["basis_file"]),
                evidence.local(ref),
                expected_active_orbitals=13,
            )["subspace"]
            row.update(
                coverage=compare["directions_at_or_above_0_95"],
                minimum_singular_value=compare["minimum_singular_value"],
            )
            row["delta_average_energy_mEh"] = (
                (row["metrics"]["energy"] - endpoints["A"]["metrics"]["energy"]) * 1000
                if label != "A"
                else 0.0
            )
            endpoints[label] = row
        mae = {
            name: float(np.mean([r["predictions"][name]["absolute_error"] for r in history[1:]]))
            for name in ("full", "no_response_updates")
        }
        require(
            round(mae["full"], 4) == 0.0358 and round(mae["no_response_updates"], 4) == 0.07,
            "Prediction MAE differs from the paper values 0.0358 and 0.0700",
        )
        require(
            [endpoints[x]["coverage"] for x in "ABC"] == [13, 12, 12],
            "A/B/C reference coverage differs from 13/12/12 directions",
        )
        require(
            [round(endpoints[x]["metrics"]["quality"], 6) for x in "ABC"]
            == [0.404431, 0.399604, 0.399603],
            "A/B/C quality scores differ from the paper values",
        )
        require(
            [round(endpoints[x]["delta_average_energy_mEh"], 3) for x in "BC"] == [4.503, 4.503],
            "B/C energy differences from A differ from 4.503 mEh",
        )
        selected = evidence.bind(
            "/home/lichen/staging/pcc_qn_fourpoints_20260926_v01/inputs/seeds/CI_CC/initial.rasscf.h5",
            "dec25a2e863a4e0b0db2fdd7f08e109e95f9c65c533c594d23120763fc9413c7",
        )
        reference = evidence.bind(
            "/home/lichen/pomdp-compute-control_20260926/supporting_files/home/lichen/staging/dioxetanone_pomdp_avas_20260911/repo/assets/dioxetanone/heldout/reference_ci_cc.dmrgscf.h5",
            "9b31f874957a890af1204ace0badcfc8020b7e3b329346efc8040ec7349faf25",
        )
        comparison = acceptance.compare_hdf5_active_subspaces(
            selected, reference, expected_active_orbitals=13
        )
        subspace = comparison["subspace"]
        with h5py.File(selected, "r") as h:
            energies = np.asarray(h["ROOT_ENERGIES"][()], float).reshape(-1)
        require(
            energies.tolist() == [-301.8246454348082, -301.82463319137935],
            "CI-CC root energies differ from the archived endpoint",
        )
        require(
            subspace["directions_at_or_above_0_95"] == 13,
            "CI-CC reference coverage is no longer 13 directions",
        )
        require(
            all((round(v, 4) == 1 for v in subspace["singular_values"])),
            "CI-CC principal singular values no longer round to 1.0000",
        )
        cicc = dict(
            scope="Initial CASSCF endpoint retained; no optimization trajectory",
            comparison=subspace,
            root_energies=energies.tolist(),
            mean_energy=float(np.mean(energies)),
        )
    evidence.unchanged()
    verify_source()
    return dict(
        source_id=source,
        numerical_baseline_source_id=SOURCE,
        physical_jobs_launched=0,
        files_unchanged=True,
        fourth_decision=fourth,
        controls=endpoints,
        prediction_history=history,
        MAE_steps_2_to_4=mae,
        MAE_reduction_percent=100 * (1 - mae["full"] / mae["no_response_updates"]),
        CI_CC=cicc,
        verified_file_count=len(bindings),
        file_bindings=bindings,
    )
