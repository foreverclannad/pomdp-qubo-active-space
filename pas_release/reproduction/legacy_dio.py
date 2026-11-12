"""Replay fixed-budget planning with regenerated composition and immutable evidence.

Derived from the frozen v18 result-reuse audit. This is an offline validator,
not a controller resume: it never writes receipts or starts physical solvers.
At the first changed dispatch in a run, stop instead of consuming evidence for
an orbital group that the counterfactual planner did not select.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from copy import deepcopy
from unittest.mock import patch
import argparse
import json
import math
import os
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True


def main():
    if not __debug__:
        raise RuntimeError('Replay verification requires Python assertions; do not use -O')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corrected-repo', type=Path, required=True)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--expected-source-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-replay', type=Path, help='Optional same-archive replay under unmodified cu07 source')
    args = parser.parse_args()
    corrected, archive = args.corrected_repo.resolve(), args.archive_root.resolve()
    if args.output.resolve().is_relative_to(archive):
        raise ValueError('Replay outputs must be outside the immutable input archive')
    if args.baseline_replay is not None and args.output.resolve() == args.baseline_replay.resolve():
        raise ValueError('The revised replay must not overwrite its baseline audit')
    sys.path.insert(0, str(corrected))
    from pas import controller, planner
    from pas.common import identity, file_hash, read_json, write_json
    from pas.config import planning_configuration
    from pas.contracts import Context
    from pas.diagnostic_window import derive_window, survey_partition
    from pas.evidence_belief import posterior_capture
    from pas.input_evidence import diagnostic_arrays
    from pas import input_evidence
    from pas.acceptance import audit as audit_acceptance
    from pas_release.common import compare_values
    from pas_release.resources import DATA
    import numpy as np

    assert Path(controller.__file__).resolve().is_relative_to(corrected)
    source_id = controller.source_identity()
    assert source_id == args.expected_source_id
    historical_comparator = read_json(DATA/'baseline.json')['files']['acceptance.py']
    def sources(repo):
        return {str(p.relative_to(repo/'pas')): file_hash(p) for p in sorted((repo/'pas').rglob('*'))
                if p.is_file() and p.suffix in ('.py', '.c')}
    old_source = read_json(DATA/'recorded_source.json')['files']
    new_source = sources(corrected)
    changed = {p: dict(original=old_source.get(p), corrected=new_source.get(p))
               for p in sorted(old_source.keys() | new_source.keys()) if old_source.get(p) != new_source.get(p)}
    original_source_id = identity(old_source)
    sidecar_root = args.output.parent/(args.output.stem + '_decisions')
    hashed = {}
    archive_hashes = {str(p.relative_to(archive)): file_hash(p) for p in sorted(archive.rglob('*')) if p.is_file()}

    def local_path(path):
        path = str(path)
        if path.startswith(('/tmp/', '/home/')):
            return archive/path.lstrip('/')
        return Path(path)

    def digest(path):
        path = local_path(path)
        if str(path) not in hashed:
            assert path.is_file(), f'Missing required archived artifact: {path}'
            hashed[str(path)] = file_hash(path)
        return hashed[str(path)]

    def binding(record):
        path, expected = record['path'], record['sha256']
        assert digest(path) == expected, f'Artifact hash mismatch: {path}'
        return dict(original_path=path, archived_path=str(local_path(path)), sha256=expected, verified=True)

    def bind_file(path):
        return dict(path=str(path), sha256=file_hash(path))

    def relocate(value):
        if isinstance(value, dict):return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):return [relocate(v) for v in value]
        if isinstance(value, str) and value.startswith(('/tmp/', '/home/')):
            p = local_path(value)
            if p.exists():return str(p)
        return value

    def receipt_checks(receipt):
        # Preserve the raw receipt identity; only file access is translated.
        with patch.object(controller, 'file_hash', digest):
            controller._verify_receipt(receipt)
        return {k: binding(v) for k, v in receipt.get('artifacts', {}).items()
                if isinstance(v, dict) and {'path', 'sha256'} <= v.keys()}

    def regenerate(context, config, saved, diagnostic, partition, density=None):
        binding(diagnostic)
        raw = read_json(local_path(diagnostic['path']))
        assert raw['active_orbital_indices'] == partition['active']
        raw_density, entropy, mutual = diagnostic_arrays(raw, context)
        if density is None:
            density = raw_density
        else:
            np.testing.assert_allclose(density, raw_density, atol=1e-12, rtol=0)
        fields = posterior_capture(context, partition, density, entropy, mutual, config)
        # The only intended numerical change is the composition proposal. Keep
        # historical observations separate from regenerated planning fields.
        for channel in ('occupation', 'qit'):
            for key in ('constant', 'linear', 'quadratic'):
                np.testing.assert_allclose(fields['channel_quadratics'][channel][key],
                    saved['channel_quadratics'][channel][key], atol=1e-12, rtol=0)
        for name, block in saved['input_posterior']['blocks'].items():
            for key in ('observation', 'prior_mean', 'mean'):
                np.testing.assert_allclose(fields['input_posterior']['blocks'][name][key], block[key], atol=1e-12, rtol=0)
            assert fields['input_posterior']['blocks'][name]['covariance'] == block['covariance']
        for key in ('context_id', 'basis_id', 'measured_indices', 'electrons', 'model_hash', 'observation_count'):
            assert fields['input_posterior'][key] == saved['input_posterior'][key]
        return fields

    def request_checks(run, event, receipt, context, config, policy):
        p = run/event['receipt']
        request_path = p.with_name('request.json')
        request = read_json(request_path)
        assert request['request_id'] == identity({k: v for k, v in request.items() if k != 'request_id'})
        assert request['request_id'] == receipt['request_id']
        partition = survey_partition(context, receipt['active']) if receipt['kind'] == 'survey' else context.partition(receipt['active'])
        expected = dict(schema='pas_physical_request_v1', context_id=context.context_id,
                        basis_id=context.basis_id,
                        checkpoint_hash=context.payload.get('checkpoint_hash', context.basis_id),
                        kind=receipt['kind'], partition=partition, protocol=config)
        assert all(identity(request[k]) == identity(v) for k, v in expected.items())
        for record in (request, receipt):
            assert record.get('fixed_dmrg_policy') == policy
            if policy is None:assert 'fixed_dmrg_policy' not in record
        assert isinstance(request['input'], str) and request['input'].strip()
        assert local_path(receipt['artifacts']['request']['path']) == request_path
        assert receipt['artifacts']['request']['sha256'] == file_hash(request_path)
        return bind_file(request_path)

    started = time.monotonic()
    result = dict(schema='pas_terminal_composition_replay_v1', checked_hkt=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
        original_source_id=original_source_id, corrected_source_id=source_id,
        corrected_repo=str(corrected), archive_root=str(archive), changed_scientific_sources=changed,
        physical_jobs_launched=0, new_physical_receipts_created=0,
        original_ledgers_modified=False, physical_costs_and_budgets_reset=False,
        semantics='Corrected planning sidecars reuse immutable original physical receipts only after identical dispatch is demonstrated.',
        reference_data_fed_to_search=False, evidence_regenerated_from_raw_diagnostic_arrays=True,
        divergence_policy='stop_each_run_at_first_changed_dispatch_no_counterfactual_output_imputation', runs={})
    names = ('lichen_pas_nx1_20261007_v18_determinant', 'lichen_pas_nx1_20261007_v18_harmonic',
             'lichen_pas_n00_20261007_v18_determinant')
    for name in names:
        run = archive/'tmp'/name/'run'
        manifest, seed, state = [read_json(run/f'{n}.json') for n in ('manifest', 'seed', 'state')]
        assert manifest['source_id'] == original_source_id
        assert manifest['manifest_id'] == identity({k: v for k, v in manifest.items() if k != 'manifest_id'})
        assert identity(seed) == manifest['seed_id'] and state['manifest_id'] == manifest['manifest_id']
        assert state['status'] == 'passed' and state['pending'] is None
        config = manifest['config']; planning = planning_configuration(config, manifest.get('response_policy'))
        initial_artifacts = receipt_checks(seed['incumbent'])
        controller._validate_metrics(seed['incumbent'], config)
        relocated_seed = relocate(seed); context = Context.from_dict(relocated_seed['context'])
        assert digest(seed['context']['payload']['basis_file']) == context.basis_id
        incumbent = seed['incumbent']; quality = incumbent['metrics']['quality']
        receipts, rows = [], []; nd, no = config['diagnostic_budget'], config['optimization_budget']
        divergence = None
        costs = dict(cpu_seconds=0.0, wall_seconds=0.0)
        bootstrap = None; bootstrap_record = None
        if 'bootstrap_input_evidence' in manifest:
            bootstrap_record = binding(manifest['bootstrap_input_evidence'])
            bootstrap_raw = read_json(local_path(manifest['bootstrap_input_evidence']['path']))
            # Bind the raw saved survey and source export, without reinterpreting it as a new event.
            bootstrap_deps = {k: binding(bootstrap_raw[k]) for k in ('source', 'diagnostic')
                              if isinstance(bootstrap_raw.get(k), dict) and {'path', 'sha256'} <= bootstrap_raw[k].keys()}
            bootstrap_record['inputs'] = bootstrap_deps
            bootstrap = relocate(bootstrap_raw)
            document = read_json(local_path(bootstrap_raw['source']['path']))
            for source in [document['diagnostic'], *document['initial_metric_sources'].values(),
                           *document['source_bindings'].values()]:
                binding(source)
            with patch.object(input_evidence, 'read_json', lambda path: read_json(local_path(path))):
                input_evidence.cross_bind(document, read_json(local_path(bootstrap_raw['diagnostic']['path'])))
            for key in ('basis_id', 'geometry_id', 'nmo', 'active', 'inactive', 'frozen', 'deleted',
                        'electrons', 'total_electrons', 'states', 'electron_roles', 'composition'):
                assert identity(document['context'][key]) == identity(seed['context'][key]), key
            assert digest(seed['context']['payload']['checkpoint']) == seed['context']['payload']['checkpoint_hash']
            bootstrap['proposal_quadratic'] = regenerate(context, config, bootstrap_raw['proposal_quadratic'],
                bootstrap_raw['diagnostic'], dict(active=bootstrap_raw['measured_window'],
                                                  electrons=document['diagnostic_electrons']))
        for event in state['events']:
            number = event['number']; assert number == len(receipts)+1
            print(json.dumps(dict(run=name, action=number, stage='planning', seed=planning['seed'],
                annealing_steps=planning['annealing_steps'], annealing_restarts=planning['annealing_restarts'],
                pool_size=planning['pool_size'])), flush=True)
            context_before = context
            policy = controller._fixed_dmrg_policy(manifest, number)
            domain, survey, bootstrap_active = controller._input_domain(context, bootstrap, receipts, fixed_dmrg_policy=policy)
            if controller._requires_context_survey(config) and survey is None:
                assert nd > 0 and not any(r['kind'] == 'survey' and r['context_id'] == context.context_id for r in receipts)
                window = derive_window(context, manifest['survey_window_policy'])
                decision = dict(kind='survey', active=window['active'], audit=dict(
                    stage='current_context_evidence_collection', context_id=context.context_id, survey_window=window))
            else:
                belief = controller._belief(context, planning, receipts, relocated_seed, bootstrap)
                excluded = {kind: [r['active'] for r in receipts if r['kind'] == kind and
                            r['context_id'] == context.context_id] for kind in ('probe', 'optimize')}
                decision = planner.choose(domain, belief, quality, nd, no, planning,
                    event_index=len(receipts), excluded=excluded, survey=survey,
                    planner_policy=manifest['planner_policy'])
                assert sorted(context.active) not in decision['pool']
                assert decision['active'] != sorted(context.active)
                if bootstrap_active:
                    decision['audit']['bootstrap_input_evidence'] = dict(source=bootstrap_raw['source'],
                        measured_window=bootstrap_raw['measured_window'], physical_window=list(context.window),
                        current_protocol_receipt=False, response_observation=False, validity_observation=False,
                        diagnostic_budget_charged=False, event_index=len(receipts))
            if policy is not None:decision['audit']['fixed_dmrg_policy'] = deepcopy(policy)
            old_decision_path = run/'decisions'/f'{number:03}.json'; old_decision = read_json(old_decision_path)
            assert old_decision['kind'] == event['kind']
            same_dispatch = (decision['kind'], decision['active']) == (old_decision['kind'], old_decision['active'])
            sidecar = dict(schema='pas_composition_replay_sidecar_v1', physical_receipt=False,
                corrected_source_id=source_id, original_source_id=original_source_id,
                context_id=context_before.context_id, basis_id=context_before.basis_id,
                evidence_prefix=[r['evidence_id'] for r in receipts],
                original_decision=bind_file(old_decision_path), decision=decision,
                original_selected=old_decision['active'], original_kind=old_decision['kind'],
                identical_original_dispatch=same_dispatch, pool_unchanged=decision.get('pool') == old_decision.get('pool'))
            if decision['kind'] == 'optimize':
                sidecar['regenerated_proposal_quadratic'] = survey['proposal_quadratic']
            sidecar_path = sidecar_root/name/f'{number:03}.json'; write_json(sidecar_path, sidecar)
            if not same_dispatch:
                divergence = dict(number=number, original_kind=old_decision['kind'], original_active=old_decision['active'],
                    new_kind=decision['kind'], new_active=decision['active'], sidecar=bind_file(sidecar_path))
                print(json.dumps(dict(run=name, divergence=divergence)), flush=True)
                break
            receipt_path = run/event['receipt']; assert file_hash(receipt_path) == event['sha256']
            receipt = read_json(receipt_path); artifacts = receipt_checks(receipt)
            assert receipt['active'] == decision['active'] and receipt['context_id'] == context.context_id
            assert receipt['evidence_id'] == event['evidence_id'] and bool(receipt['valid']) == event['valid']
            request_record = request_checks(run, event, receipt, context, config, policy)
            controller._survey_posterior(receipt, context, config)
            if receipt['kind'] != 'survey':controller._validate_metrics(receipt, config)
            accepted = bool(receipt['kind'] == 'optimize' and receipt['valid'] and
                            receipt['metrics']['quality'] > quality + config['acceptance_tolerance'])
            assert accepted == event['accepted']
            if accepted:
                context = controller._optimized_output(context, relocate(receipt))
                incumbent, quality = receipt, receipt['metrics']['quality']
                assert digest(receipt['output_context']['payload']['basis_file']) == context.basis_id
            for key in costs:
                assert math.isfinite(receipt[key]) and receipt[key] >= 0
                costs[key] += receipt[key]
            if receipt['kind'] == 'optimize':no -= 1
            else:nd -= 1
            assert min(nd, no) >= 0
            sidecar['original_receipt'] = bind_file(receipt_path)
            planning_receipt = relocate(receipt)
            if receipt['kind'] == 'survey' and receipt['valid']:
                density = np.load(local_path(receipt['artifacts']['densities']['path']), allow_pickle=False)
                fields = regenerate(context_before, config, receipt['proposal_quadratic'],
                    receipt['artifacts']['mps_evidence'], survey_partition(context_before, receipt['active']), density)
                sidecar['regenerated_proposal_quadratic'] = fields
                planning_receipt['proposal_quadratic'] = fields
                planning_receipt['physical_receipt'] = False
            write_json(sidecar_path, sidecar)
            rows.append(dict(number=number, kind=receipt['kind'], selected=decision['active'],
                context_id=context_before.context_id, next_context_id=context.context_id,
                corrected_decision=bind_file(sidecar_path), original_decision=bind_file(old_decision_path),
                original_receipt=bind_file(receipt_path), request=request_record, request_id=receipt['request_id'],
                evidence_id=receipt['evidence_id'], accepted=accepted, valid=receipt['valid'],
                metrics=receipt.get('metrics'), reused_artifacts=artifacts,
                original_physical_cost={k: receipt[k] for k in costs}, new_physical_cost=0,
                remaining_diagnostics=nd, remaining_optimizations=no))
            receipts.append(planning_receipt)
            print(json.dumps(dict(run=name, action=number, reused=True, accepted=accepted)), flush=True)
        if divergence is not None:
            result['runs'][name] = dict(original_manifest=bind_file(run/'manifest.json'),
                original_seed=bind_file(run/'seed.json'), original_state=bind_file(run/'state.json'),
                bootstrap_input=bootstrap_record, decisions=rows, divergence=divergence,
                all_original_dispatches_reproduced=False, acceptance_bound_to_same_physical_files=False,
                status='stopped_at_first_changed_dispatch', new_physical_cost=0)
            write_json(args.output, result)
            continue
        assert identity(incumbent) == identity(state['incumbent'])
        assert context.context_id == Context.from_dict(state['context']).context_id
        assert nd == state['remaining_diagnostics'] and no == state['remaining_optimizations']
        assert all(math.isclose(costs[k], state['search_'+k], rel_tol=0, abs_tol=1e-9) for k in costs)
        acceptance_path = run/'acceptance.json'; acceptance = read_json(acceptance_path)
        assert acceptance == state['acceptance'] and acceptance['passed']
        assert acceptance['incumbent_evidence_id'] == incumbent['evidence_id']
        assert acceptance['incumbent_sha256'] == identity(incumbent)
        # Authenticate the historical implementation separately from this release.
        # Relocated copies are numerical comparison views, never new receipts.
        refs = binding(manifest['references'])
        current_acceptance = audit_acceptance(
            relocate(incumbent), relocate(read_json(local_path(manifest['references']['path']))))
        assert current_acceptance['passed'] == acceptance['passed']
        assert current_acceptance['comparator_sha256'] == file_hash(corrected/'pas/acceptance.py')
        assert current_acceptance['comparisons'].keys() == acceptance['comparisons'].keys()
        reference_errors = {}
        coverage = {}
        for label, comparison in acceptance['comparisons'].items():
            assert comparison['status'] == 'complete'
            subspace = comparison['subspace']
            selected_binding, reference_binding = binding(comparison['selected']), binding(comparison['reference'])
            fresh = comparison['fresh_binding']
            assert fresh['candidate_sha256'] == incumbent['artifacts']['basis_file']['sha256'] == selected_binding['sha256']
            assert fresh['reference_sha256'] == reference_binding['sha256']
            assert acceptance['comparator_sha256'] == historical_comparator == fresh['comparator_code_sha256']
            current_subspace = current_acceptance['comparisons'][label]['subspace']
            # Near-unit arccos angles are diagnostic; compare the stable singular
            # spectrum and coverage statistics at the existing 1e-12 tolerance.
            angles = {'principal_angles_radians', 'principal_angles_degrees', 'maximum_principal_angle_degrees'}
            reference_errors[label] = compare_values(
                {k: v for k, v in current_subspace.items() if k not in angles},
                {k: v for k, v in subspace.items() if k not in angles},
                location=name+'.'+label)
            assert acceptance['required_directions'] == subspace['dimension'] == 13
            assert subspace['directions_at_or_above_0_95'] == 13 and subspace['minimum_singular_value'] >= .95
            coverage[label] = dict(directions=13, dimension=13, minimum_singular_value=subspace['minimum_singular_value'],
                                   candidate=selected_binding, reference=reference_binding)
        assert coverage
        assert acceptance['reference_data_fed_to_search'] is False
        result['runs'][name] = dict(original_manifest=bind_file(run/'manifest.json'),
            original_seed=bind_file(run/'seed.json'), original_state=bind_file(run/'state.json'),
            initial_artifacts=initial_artifacts, original_initialization=seed['initialization'],
            bootstrap_input=bootstrap_record, decisions=rows, all_original_dispatches_reproduced=True,
            final_status=state['status'], final_evidence_id=incumbent['evidence_id'], final_metrics=incumbent['metrics'],
            final_original_artifacts=receipt_checks(incumbent), original_reference_manifest=refs,
            original_acceptance=bind_file(acceptance_path), acceptance_bound_to_same_physical_files=True,
            coverage=coverage, reference_comparison_max_errors=reference_errors,
            historical_comparator_sha256=historical_comparator,
            current_comparator_sha256=current_acceptance['comparator_sha256'],
            search_costs_preserved=costs, remaining_diagnostics=nd, remaining_optimizations=no,
            original_initialization_costs_preserved=True, new_physical_cost=0)
        write_json(args.output, result)
    assert sources(corrected) == new_source
    assert {str(p.relative_to(archive)): file_hash(p) for p in sorted(archive.rglob('*')) if p.is_file()} == archive_hashes
    result['archive_files_unchanged'] = len(archive_hashes)
    result['verified_required_artifact_files'] = len(hashed)
    result['all_original_dispatches_reproduced'] = all(run['all_original_dispatches_reproduced'] for run in result['runs'].values())
    result['all_five_optimization_choices_reproduced'] = result['all_original_dispatches_reproduced'] and sum(r['kind'] == 'optimize' for run in result['runs'].values() for r in run['decisions']) == 5
    result['all_three_original_13_of_13_results_reusable'] = result['all_original_dispatches_reproduced']
    result['elapsed_seconds'] = time.monotonic() - started
    result['runtime'] = dict(python=sys.version, numpy=np.__version__)
    if args.baseline_replay is not None:
        baseline = read_json(args.baseline_replay)
        assert baseline['original_source_id'] == original_source_id and baseline['all_original_dispatches_reproduced']
        comparison = dict(baseline_audit=bind_file(args.baseline_replay.resolve()),
            baseline_source_id=baseline['corrected_source_id'], actions=[])
        for name, run in result['runs'].items():
            for row in run['decisions']:
                previous = next(r for r in baseline['runs'][name]['decisions'] if r['number'] == row['number'])
                old_binding, new_binding = previous['corrected_decision'], row['corrected_decision']
                assert file_hash(old_binding['path']) == old_binding['sha256']
                assert file_hash(new_binding['path']) == new_binding['sha256']
                old, new = [read_json(b['path'])['decision'] for b in (old_binding, new_binding)]
                assert previous['context_id'] == row['context_id']
                comparison['actions'].append(dict(run=name, number=row['number'], kind=row['kind'],
                    selected=row['selected'], identical_dispatch=(old['kind'], old['active']) == (new['kind'], new['active']),
                    baseline_pool=old.get('pool', []), terminal_pool=new.get('pool', []),
                    pool_unchanged=old.get('pool') == new.get('pool'), baseline_decision=old_binding,
                    terminal_decision=new_binding))
        result['baseline_comparison'] = comparison
    result['audit_script'] = bind_file(Path(__file__).resolve())
    write_json(args.output, result)
    print(json.dumps(dict(output=str(args.output), sha256=file_hash(args.output), complete=True)), flush=True)


if __name__ == '__main__':
    main()
