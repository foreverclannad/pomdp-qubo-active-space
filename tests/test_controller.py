# Selected from c06614a test_controller.py; retention rationale: docs/maintenance/retention.json
"""State-machine tests use real posterior/planner contracts but no chemistry."""

import tempfile
import unittest
from pathlib import Path

from physical_fixture import bind_request

from pas.common import file_hash, identity, read_json, write_json
from pas.config import configuration
from pas.contracts import Context
from pas.controller import prepare, run


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = configuration(
            dict(backend="pyscf", diagnostic_budget=1, optimization_budget=2, threads=1)
        )
        self.ctx = Context(
            basis_id="initial",
            geometry_id="molecule",
            backend="pyscf",
            nmo=4,
            active=(1, 2),
            inactive=(0,),
            total_electrons=4,
            electrons=2,
            composition=[0.2, 0.4, 0.6, 0.8],
            effective_config_id=identity(self.cfg),
        )
        self.initial = self.receipt(self.ctx, [1, 2], "initial", 0.3)
        self.seed = dict(context=self.ctx.to_dict(), incumbent=self.initial, initialization={})

    def tearDown(self):
        self.temp.cleanup()

    def receipt(self, ctx, active, kind, value, valid=True):
        r = dict(
            context_id=ctx.context_id,
            request_id=identity([active, kind, value, ctx.context_id]),
            kind=kind,
            active=list(active),
            valid=valid,
            converged=valid,
            state_valid=valid,
            metrics=dict(occupation=value, composition=value, quality=value) if valid else {},
            cpu_seconds=2.0,
            wall_seconds=1.0,
            output_context=ctx.to_dict(),
        )
        r["evidence_id"] = identity(r)
        if kind != "initial":
            pending = read_json(self.root / "state.json")["pending"]
            bind_request(
                r,
                ctx,
                self.root / pending["directory"],
                read_json(self.root / "manifest.json")["config"],
            )
        return r

    def resign(self, receipt):
        receipt["evidence_id"] = identity({k: v for k, v in receipt.items() if k != "evidence_id"})
        return receipt

    def bind_seed_config(self, config):
        self.ctx.effective_config_id = identity(config)
        if "composition" not in config["channels"]:
            self.ctx.composition = None
        self.seed["context"] = self.ctx.to_dict()
        self.initial.update(output_context=self.ctx.to_dict(), context_id=self.ctx.context_id)
        values = {k: self.initial["metrics"][k] for k in config["channels"]}
        values["quality"] = sum(
            (config["weights"][k] * values[k] for k in config["channels"])
        ) / sum(config["weights"].values())
        self.initial["metrics"] = values
        self.resign(self.initial)

    def reserve_interrupted_request(self):
        prepare(self.seed, self.cfg, self.root)

        def crash(*args, **kwargs):
            raise OSError("lost transport")

        with self.assertRaises(OSError):
            run(
                self.root,
                execute=crash,
                choose=lambda *a, **kw: dict(kind="optimize", active=[0, 2]),
            )
        state = read_json(self.root / "state.json")
        return self.root / state["pending"]["directory"]

    def test_rejected_then_accepted_budget_and_new_context(self):
        prepare(self.seed, self.cfg, self.root)
        seen = []

        def choose(ctx, b, u, d, h, c, **kw):
            seen.append((ctx.context_id, dict(b.validity)))
            return dict(kind="optimize", active=[0, 2] if h == 2 else [0, 3])

        def execute(ctx, active, kind, directory, config):
            r = self.receipt(ctx, active, kind, 0.2 if active == [0, 2] else 0.7)
            if active == [0, 3]:
                new = Context.from_dict(ctx.to_dict())
                new.basis_id = "optimized"
                r["output_context"] = new.to_dict()
                self.resign(r)
            return r

        result = run(self.root, execute=execute, choose=choose)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["remaining_optimizations"], 0)
        self.assertEqual(result["remaining_diagnostics"], 1)
        self.assertEqual([e["accepted"] for e in result["events"]], [False, True])
        self.assertEqual(seen[1][1]["optimize"], [2, 1])
        self.assertEqual(result["search_cpu_seconds"], 4)
        self.assertEqual(result["context"]["basis_id"], "optimized")
        updates = [read_json(self.root / "filter_updates" / f"{i:03d}.json") for i in (1, 2)]
        for update, value in zip(updates, (0.2, 0.7)):
            self.assertEqual(update["context_id"], self.ctx.context_id)
            self.assertTrue(update["target_quality_updated"])
            self.assertEqual(update["role"], "conditioned_input_context")
            for channel in self.cfg["channels"]:
                predicted = update["predictions_at_observed_candidate"][channel]
                self.assertAlmostEqual(predicted["mean"][1], value, places=8)
                self.assertLess(predicted["covariance"][1][1], 1e-09)
                self.assertEqual(update["jitter"][channel], 1e-12)
        self.assertFalse(updates[0]["accepted_output_starts_fresh_quality_prior"])
        self.assertTrue(updates[1]["accepted_output_starts_fresh_quality_prior"])
        self.assertEqual(updates[1]["observations"], dict(occupation=2, composition=2))
        new_context = Context.from_dict(result["context"])
        self.assertEqual(updates[1]["next_context_id"], new_context.context_id)
        from pas.controller import _belief, _read_events

        new_belief = _belief(new_context, self.cfg, _read_events(self.root, result), self.seed)
        self.assertTrue(all((not rows for rows in new_belief.observations.values())))
        self.assertEqual(new_belief.validity["optimize"], [3, 1])

    def test_pending_receipt_is_committed_once_without_dispatch(self):
        directory = self.reserve_interrupted_request()
        receipt = self.receipt(self.ctx, [0, 2], "optimize", 0.2)
        write_json(directory / "receipt.json", receipt)

        def forbidden(*a, **kw):
            self.fail("resume dispatched a physical request")

        for _ in range(2):
            result = run(self.root, execute=forbidden, choose=forbidden, max_actions=0)
            self.assertIsNone(result["pending"])
            self.assertEqual(len(result["events"]), 1)
            self.assertEqual(result["remaining_optimizations"], 1)
            self.assertEqual(result["search_cpu_seconds"], 2)

    def test_request_is_rechecked_when_replaying_a_completed_ledger(self):
        prepare(self.seed, self.cfg, self.root)
        result = run(
            self.root,
            choose=lambda *a, **kw: dict(kind="optimize", active=[0, 2]),
            execute=lambda c, a, k, d, x: self.receipt(c, a, k, 0.2, valid=False),
            max_actions=1,
        )
        path = self.root / result["events"][0]["receipt"]
        receipt = read_json(path)
        request_path = path.parent / "request.json"
        request = read_json(request_path)
        request["protocol"]["threads"] = 2
        request["request_id"] = identity({k: v for k, v in request.items() if k != "request_id"})
        write_json(request_path, request)
        receipt["request_id"] = request["request_id"]
        receipt["artifacts"]["request"]["sha256"] = file_hash(request_path)
        self.resign(receipt)
        write_json(path, receipt)
        result["events"][0].update(evidence_id=receipt["evidence_id"], sha256=file_hash(path))
        write_json(self.root / "state.json", result)
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            run(self.root, execute=lambda *a: self.fail("bad ledger dispatched"), max_actions=0)

    def test_invalid_cost_cannot_partially_commit_an_improvement(self):
        directory = self.reserve_interrupted_request()
        receipt = self.receipt(self.ctx, [0, 2], "optimize", 0.7)
        for bad_cost in (-1, None):
            with self.subTest(cost=bad_cost):
                if bad_cost is None:
                    receipt.pop("cpu_seconds")
                else:
                    receipt["cpu_seconds"] = bad_cost
                self.resign(receipt)
                write_json(directory / "receipt.json", receipt)
                with self.assertRaisesRegex(ValueError, "physical cost|Physical cost"):
                    run(self.root, max_actions=0)
                state = read_json(self.root / "state.json")
                self.assertEqual(state["incumbent"]["evidence_id"], self.initial["evidence_id"])
                self.assertEqual(state["events"], [])
                self.assertEqual(state["search_cpu_seconds"], 0)

    def test_pending_unknown_is_not_reexecuted(self):
        prepare(self.seed, self.cfg, self.root)

        def choose(*args, **kwargs):
            return dict(kind="optimize", active=[0, 2])

        def crash(*args, **kwargs):
            raise OSError("lost transport")

        with self.assertRaises(OSError):
            run(self.root, execute=crash, choose=choose)
        state = read_json(self.root / "state.json")
        self.assertEqual(state["remaining_optimizations"], 1)
        with self.assertRaisesRegex(RuntimeError, "Unresolved"):
            run(self.root, execute=lambda *a: self.fail("reran unknown request"), choose=choose)

    def test_invalid_result_spends_budget_and_only_updates_validity(self):
        cfg = configuration(dict(backend="pyscf", diagnostic_budget=0, optimization_budget=2))
        self.bind_seed_config(cfg)
        prepare(self.seed, cfg, self.root)

        def choose(ctx, b, *args, **kwargs):
            n = kwargs["event_index"]
            if n:
                self.assertEqual(b.validity["optimize"], [1, 2])
            return dict(kind="optimize", active=[0, 2] if not n else [0, 3])

        result = run(
            self.root,
            choose=choose,
            execute=lambda c, a, k, d, x: self.receipt(c, a, k, 0.2, valid=False),
        )
        self.assertEqual(result["incumbent"]["evidence_id"], self.initial["evidence_id"])
        self.assertEqual(result["remaining_optimizations"], 0)
        update = read_json(self.root / "filter_updates/002.json")
        self.assertFalse(update["target_quality_updated"])
        self.assertTrue(update["validity_updated"])
        self.assertEqual(update["validity_beta"]["optimize"], [1, 3])
        self.assertEqual(update["observations"], dict(occupation=0, composition=0))


if __name__ == "__main__":
    unittest.main()
