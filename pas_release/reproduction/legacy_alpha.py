"""Offline alpha sensitivity; immutable observations, no physical dispatch.

Use the public configuration parameter and its new context-derived random seeds.
Stop a trajectory at its first changed dispatch or acceptance. Rescored copies
are planning-only objects, never newly signed physical receipts.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alphas', type=float, nargs='+', required=True)
    parser.add_argument('--tif3-root', type=Path)
    parser.add_argument('--extra-archive', type=Path)
    parser.add_argument('--cases', nargs='+', default=['NX1', 'CI_OO_N00', 'TiF3_first'])
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a fresh result path'
    assert all(0 <= a <= 1 for a in args.alphas)
    assert args.alphas[0] == .5, 'First reproduce the unchanged baseline'
    sys.path.insert(0, str(args.repo))
    import numpy as np
    from pas import metrics, evidence_belief, controller, planner, qubo
    from pas.common import file_hash, identity, read_json
    from pas.config import configuration, planning_configuration
    from pas.contracts import Context
    from pas.diagnostic_window import derive_window, survey_partition
    from pas.input_evidence import diagnostic_arrays

    archive = args.archive.resolve()
    bindings = {}
    alpha = .5
    source_id = controller.source_identity()

    def local(path):
        path = str(path)
        if path.startswith(('/tmp/', '/home/', '/localdata/')):
            for base in [archive, args.extra_archive]:
                if base is not None:
                    candidate = base / path.lstrip('/')
                    if candidate.exists():
                        return candidate
        return Path(path)

    def read(path, expected=None):
        path = local(path)
        digest = file_hash(path)
        if expected is not None:
            assert digest == expected, str(path)
        bindings[str(path)] = digest
        return read_json(path)

    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, str) and value.startswith(('/tmp/', '/home/', '/localdata/')):
            path = local(value)
            if path.exists():
                return str(path)
        return value

    def components(context, active):
        return metrics.composition_components(context.composition, active,
            context.composition_pair, 'determinant_pair_v1',
            composition_channels=context.composition_channels)

    def rescore(raw, context, config):
        row = relocate(copy.deepcopy(raw))
        if row.get('valid') and 'metrics' in row and 'composition' in row['metrics']:
            p, b = components(context, context.active)
            assert np.isclose(raw['metrics']['composition'], (p + b) / 2, atol=1e-12, rtol=0)
            row['metrics']['composition'] = metrics.composition_score(context.composition, context.active,
                context.composition_pair, config['composition_model'],
                composition_purity_weight=config.get('composition_purity_weight', .5))
            row['metrics']['quality'] = sum(config['weights'][c] * row['metrics'][c]
                for c in config['channels']) / sum(config['weights'].values())
        return row

    def regenerate(raw, context, config, bootstrap=False):
        row = relocate(copy.deepcopy(raw))
        binding = raw['diagnostic'] if bootstrap else raw['artifacts']['mps_evidence']
        diagnostic = read(binding['path'], binding['sha256'])
        density, entropy, mutual = diagnostic_arrays(diagnostic, context)
        active = raw['measured_window'] if bootstrap else raw['active']
        fields = evidence_belief.posterior_capture(context, survey_partition(context, active),
            density, entropy, mutual, config)
        saved = raw['proposal_quadratic']
        assert fields['input_posterior']['blocks'] == saved['input_posterior']['blocks']
        for channel in ('occupation', 'qit'):
            for key in ('constant', 'linear', 'quadratic'):
                np.testing.assert_allclose(fields['channel_quadratics'][channel][key],
                    saved['channel_quadratics'][channel][key], atol=1e-12, rtol=0)
        # Verify exact quadratic composition for more than the incumbent.
        quad = fields['channel_quadratics']['composition']
        for active_test in [context.active, active[:context.k]]:
            x = np.zeros(context.nmo)
            x[list(active_test)] = 1
            score = quad['constant'] + np.asarray(quad['linear']) @ x + x @ np.asarray(quad['quadratic']) @ x
            assert abs(score - metrics.composition_score(context.composition, active_test,
                context.composition_pair, 'determinant_pair_v1', composition_purity_weight=alpha)) < 1e-12
        row['proposal_quadratic'] = fields
        return row

    runs = {}
    for label, folder in [('NX1', 'lichen_pas_nx1_20261007_v18_determinant'),
                          ('CI_OO_N00', 'lichen_pas_n00_20261007_v18_determinant')]:
        if label not in args.cases:
            continue
        root = archive / 'tmp' / folder / 'run'
        manifest, seed, state = [read(root / f'{name}.json') for name in ('manifest', 'seed', 'state')]
        assert identity(seed) == manifest['seed_id']
        assert manifest['manifest_id'] == identity({k: v for k, v in manifest.items() if k != 'manifest_id'})
        assert state['manifest_id'] == manifest['manifest_id']
        bootstrap = read(manifest['bootstrap_input_evidence']['path'],
                         manifest['bootstrap_input_evidence']['sha256']) if 'bootstrap_input_evidence' in manifest else None
        events = []
        for event in state['events']:
            receipt = read(root / event['receipt'], event['sha256'])
            assert receipt['evidence_id'] == identity({k: v for k, v in receipt.items() if k != 'evidence_id'})
            decision = read(root / 'decisions' / f"{event['number']:03}.json")
            events.append((event, receipt, decision))
        runs[label] = dict(manifest=manifest, seed=seed, state=state, bootstrap=bootstrap, events=events)

    if args.tif3_root and 'TiF3_first' in args.cases:
        root = args.tif3_root
        manifest, seed = [read(root / f'{name}.json') for name in ('manifest', 'seed')]
        assert identity(seed) == manifest['seed_id']
        events = []
        for number, kind in ((1, 'survey'), (2, 'optimize')):
            receipt = read(root / 'actions' / f'{number:03}_{kind}' / 'receipt.json')
            assert receipt['evidence_id'] == identity({k: v for k, v in receipt.items() if k != 'evidence_id'})
            decision = read(root / 'decisions' / f'{number:03}.json')
            event = dict(number=number, kind=kind, accepted=False)
            events.append((event, receipt, decision))
        runs['TiF3_first'] = dict(manifest=manifest, seed=seed, state=None, bootstrap=None, events=events)

    result = dict(schema='configured_alpha_replay_v1', source_id=source_id,
        script_sha256=file_hash(__file__), composition='alpha*P+(1-alpha)*B',
        proposal_qit='original_window_normalization_unchanged', terminal_qit='original_K_pair_ln4_unchanged',
        seed_policy='formal_config_identity_and_context_derived_rng_seeds',
        scope='finite_parameter_grid_not_continuous_interval_proof',
        trajectory_policy='stop_at_first_changed_dispatch_or_acceptance',
        physical_jobs_launched=0, original_files_modified=False, points=[], acceptance_lines={})
    for label, data in runs.items():
        context = Context.from_dict(relocate(data['seed']['context']))
        cfg = planning_configuration(data['manifest']['config'], data['manifest'].get('response_policy'))
        previous_metrics = data['seed']['incumbent']['metrics']
        pp, pb = components(context, context.active)
        lines = []
        for event, receipt, _ in data['events']:
            if receipt['kind'] != 'optimize' or not receipt.get('valid'):
                continue
            out = Context.from_dict(relocate(receipt['output_context']))
            p, b = components(out, out.active)
            w, total = cfg['weights'], sum(cfg['weights'].values())
            intercept = (sum(w[c] * (receipt['metrics'][c] - previous_metrics[c])
                for c in cfg['channels'] if c != 'composition') + w['composition'] * (b - pb)) / total
            slope = w['composition'] * ((p - pp) - (b - pb)) / total
            lines.append(dict(action=event['number'], intercept=intercept, slope=slope,
                tolerance=cfg['acceptance_tolerance'], original_accepted=event['accepted'],
                equality_alpha=(cfg['acceptance_tolerance'] - intercept) / slope if slope else None))
            if event['accepted']:
                pp, pb, previous_metrics = p, b, receipt['metrics']
        result['acceptance_lines'][label] = lines

    baseline_pools = {}
    started = time.monotonic()
    for value in args.alphas:
        alpha = value
        point = dict(alpha=alpha, runs={})
        for label, data in runs.items():
            t0 = time.monotonic()
            manifest = data['manifest']
            physical_cfg = configuration({**manifest['config'], 'composition_purity_weight': alpha})
            cfg = planning_configuration(physical_cfg, manifest.get('response_policy'))
            context_ids = {}
            for original in [data['seed']['context']] + [
                    r['output_context'] for _, r, _ in data['events'] if 'output_context' in r]:
                old_context = Context.from_dict(original)
                revised = Context.from_dict({**original, 'effective_config_id': identity(physical_cfg)})
                context_ids[old_context.context_id] = revised.context_id

            def rebind(value):
                if isinstance(value, dict):
                    return {k: identity(physical_cfg) if k == 'effective_config_id' else rebind(v)
                            for k, v in value.items()}
                if isinstance(value, list):
                    return [rebind(v) for v in value]
                return context_ids.get(value, value) if isinstance(value, str) else value

            assert cfg['composition_model'] == 'determinant_pair_v1'
            assert 'proposal_score_model' not in cfg
            seed = rebind(relocate(copy.deepcopy(data['seed'])))
            context = Context.from_dict(seed['context'])
            seed['incumbent'] = rebind(rescore(data['seed']['incumbent'], context, cfg))
            quality = seed['incumbent']['metrics']['quality']
            boot = rebind(regenerate(data['bootstrap'], context, cfg, True)) if data['bootstrap'] else None
            nd, no = cfg['diagnostic_budget'], cfg['optimization_budget']
            receipts, rows = [], []
            divergence = None
            for event, raw, old in data['events']:
                number = event['number']
                policy = controller._fixed_dmrg_policy(manifest, number)
                domain, survey, _ = controller._input_domain(context, boot, receipts, fixed_dmrg_policy=policy)
                if controller._requires_context_survey(cfg) and survey is None:
                    if label == 'TiF3_first':
                        # TiF3 analysis is conditional on the original survey.
                        # Its window policy is alpha-independent; no new HDF5
                        # geometry/window derivation is claimed in this audit.
                        assert raw['kind'] == 'survey' and context_ids[raw['context_id']] == context.context_id
                        assert raw['valid'] and raw['converged'] and raw['state_valid']
                        decision = dict(kind='survey', active=raw['active'])
                    else:
                        window = derive_window(context, manifest['survey_window_policy'])
                        decision = dict(kind='survey', active=window['active'])
                else:
                    belief = controller._belief(context, cfg, receipts, seed, boot)
                    excluded = {kind: [r['active'] for r in receipts if r['kind'] == kind
                        and r['context_id'] == context.context_id] for kind in ('probe', 'optimize')}
                    decision = planner.choose(domain, belief, quality, nd, no, cfg,
                        event_index=len(receipts), excluded=excluded, survey=survey,
                        planner_policy=manifest['planner_policy'])
                matched = (decision['kind'], decision['active']) == (old['kind'], old['active'])
                row = dict(action=number, kind=decision['kind'], selected=decision['active'],
                    original_selected=old['active'], dispatch_unchanged=matched,
                    pool=decision.get('pool'), original_pool=old.get('pool'))
                if decision['kind'] == 'optimize':
                    row['feasible_count'] = qubo.candidate_count(domain)
                    row['planner_audit'] = decision.get('audit')
                    key = (label, number)
                    if alpha == .5:
                        baseline_pools[key] = decision['pool']
                    row['pool_unchanged_from_alpha_half'] = decision['pool'] == baseline_pools.get(key)
                rows.append(row)
                if not matched:
                    divergence = dict(action=number, reason='changed_dispatch')
                    break
                if raw['kind'] == 'survey':
                    receipt = rebind(regenerate(raw, context, cfg))
                    nd -= 1
                else:
                    out = Context.from_dict(rebind(relocate(raw['output_context']))) if raw.get('valid') else context
                    receipt = rebind(rescore(raw, out, cfg))
                    delta = receipt['metrics']['quality'] - quality if raw.get('valid') else None
                    accepted = bool(raw.get('valid') and delta > cfg['acceptance_tolerance'])
                    row.update(delta_quality=delta, accepted=accepted, original_accepted=event['accepted'])
                    no -= 1
                    if accepted != event['accepted']:
                        divergence = dict(action=number, reason='changed_acceptance')
                        break
                    if accepted:
                        context = out
                        quality = receipt['metrics']['quality']
                receipts.append(receipt)
            point['runs'][label] = dict(config=cfg, effective_config_id=identity(physical_cfg),
                rows=rows, divergence=divergence,
                all_original_dispatches_and_acceptances_unchanged=divergence is None,
                elapsed_seconds=time.monotonic()-t0)
            if alpha == .5:
                assert divergence is None, ('Unmodified baseline failed', label, divergence)
            print(json.dumps(dict(alpha=alpha, run=label, divergence=divergence,
                seconds=round(time.monotonic()-t0, 3))), flush=True)
        result['points'].append(point)
        result['elapsed_seconds'] = time.monotonic()-started
        result['input_bindings'] = bindings
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    assert all(file_hash(path) == digest for path, digest in bindings.items())
    assert controller.source_identity() == source_id
    result['inputs_unchanged_verified'] = True
    result['runtime'] = dict(python=sys.version, numpy=np.__version__)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
