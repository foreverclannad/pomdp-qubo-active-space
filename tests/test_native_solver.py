# Selected from c06614a test_configurable_states_native.py; retention rationale: docs/maintenance/retention.json
"""Real multi-root checkpoints and QIT for configurable states, versus FCI."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pas import metrics
from pas.common import identity, write_json
from pas.solvers.block2_checkpoint import REQUEST_SCHEMA, extract_checkpoint_entanglement
from pas.solvers.block2_convergence import diagnostic_config


@pytest.mark.integration
@pytest.mark.parametrize("spin,nroots", [(0, 2), (1, 5)])
def test_native_configured_roots_density_energy_entropy_match_fci(tmp_path, nroots, spin):
    pytest.importorskip("pyblock2.driver.core")
    pytest.importorskip("pyscf")
    if os.name != "posix":
        pytest.skip("Native Block2 executable requires Linux")
    from pyscf import ao2mo, fci, lib, tools

    lib.num_threads(1)
    driver = Path(sys.executable).parent / "block2main"
    if not driver.is_file():
        pytest.skip("Native Block2 executable is unavailable")
    sites, electrons = (6, 4 + spin % 2)
    h1 = np.diag(np.arange(sites) * 0.137)
    for i in range(sites - 1):
        h1[i, i + 1] = h1[i + 1, i] = -1.0
    eri = np.zeros((sites,) * 4)
    for i in range(sites):
        eri[i, i, i, i] = 2.0 + 0.11 * i
    packed = ao2mo.restore(8, eri, sites)
    tools.fcidump.from_integrals(str(tmp_path / "FCIDUMP"), h1, packed, sites, electrons, ms=spin)
    weights = (np.arange(1, nroots + 1) / sum(range(1, nroots + 1))).tolist()
    ids = [f"root_{i + 1}" for i in range(nroots)]
    states = dict(ids=ids, roots=list(range(1, nroots + 1)), weights=weights)
    text = diagnostic_config(
        dict(electrons=electrons, spin=spin, states=states), 64, maxiter=50, threads=1
    )
    (tmp_path / "dmrg.conf").write_text(text)
    write_json(tmp_path / "state_sweep_gate_request.json", dict(profile="standard"))
    code = 'import sys; from pas import backend; backend.BLOCK2MAIN=sys.argv[1]; from pas.solvers.diagnostic_sweep_gate import main; sys.argv=[backend.BLOCK2MAIN,"dmrg.conf"]; main()'
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    with (tmp_path / "dmrg.out").open("w") as log:
        job = subprocess.run(
            [sys.executable, "-c", code, str(driver)],
            cwd=tmp_path,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    assert job.returncode == 0, (tmp_path / "dmrg.out").read_text()[-5000:]
    request = dict(
        schema=REQUEST_SCHEMA,
        source_directory=str(tmp_path),
        orbital_basis_id="toy_integrals",
        physical_evidence_id="native_configured_state_regression",
        active_orbital_indices=list(range(sites)),
        active_electrons=electrons,
        spin=spin,
        state_ids=ids,
        state_weights=weights,
        checkpoint=dict(symmetry="SU2", mps_tag="KET", threads=1, stack_memory_bytes=512000000),
        requested_evidence=["occupation", "single_orbital_entropy", "mutual_information"],
        extraction_strategies=["checkpoint_mps"],
    )
    request["request_id"] = identity(request)
    result = extract_checkpoint_entanglement(request, temporary_directory=tmp_path / "extract")
    write_json(tmp_path / "result.json", result)
    solver = fci.direct_spin1.FCI()
    nelec = ((electrons + spin) // 2, (electrons - spin) // 2)
    shape = tuple((fci.cistring.num_strings(sites, n) for n in nelec))
    addresses, hamiltonian = solver.pspace(h1, packed, sites, nelec, np=int(np.prod(shape)))
    values, columns = np.linalg.eigh(hamiltonian)
    exact_energies, vectors = ([], [])
    for energy, column in zip(values, columns.T, strict=True):
        vector = np.empty(int(np.prod(shape)))
        vector[addresses] = column
        vector = vector.reshape(shape)
        ss, _ = solver.spin_square(vector, sites, nelec)
        if abs(ss - spin * (spin + 2) / 4) < 1e-08:
            exact_energies.append(energy)
            vectors.append(vector)
        if len(vectors) == nroots:
            break
    assert len(vectors) == nroots
    exact_d, exact_s = ([], [])
    for vector in vectors:
        da, db = solver.make_rdm1s(
            vector, sites, ((electrons + spin) // 2, (electrons - spin) // 2)
        )
        _, d2 = solver.make_rdm12(vector, sites, ((electrons + spin) // 2, (electrons - spin) // 2))
        double = np.einsum("iiii->i", d2) / 2
        pa, pb = (da.diagonal(), db.diagonal())
        probabilities = np.array([1 - pa - pb + double, pa - double, pb - double, double]).T
        exact_d.append(da + db)
        exact_s.append(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1))
    density = np.asarray(result["state_operators"])
    entropy = np.asarray([row["single_orbital_entropy"] for row in result["states"]])
    mutual = np.asarray([row["mutual_information"] for row in result["states"]])
    np.testing.assert_allclose(result["root_energies"], exact_energies, atol=1e-07, rtol=0)
    np.testing.assert_allclose(density, exact_d, atol=0.001, rtol=0)
    np.testing.assert_allclose(entropy, exact_s, atol=0.001, rtol=0)
    metrics.validate_densities(density, weights, electrons)
    metrics.qit(entropy, mutual, weights)
    write_json(
        tmp_path / "validation.json",
        dict(
            passed=True,
            nroots=nroots,
            spin=spin,
            weights=weights,
            energy_max_error=float(
                np.max(abs(np.asarray(result["root_energies"]) - exact_energies))
            ),
            density_max_error=float(np.max(abs(density - exact_d))),
            entropy_max_error=float(np.max(abs(entropy - exact_s))),
            tolerances=dict(energy=1e-07, density=0.001, entropy=0.001, projection_density=1e-08),
            checkpoint_unchanged=result["source_checkpoint_unchanged"],
        ),
    )
    assert result["state_weights"] == weights and result["source_checkpoint_unchanged"]
    assert result["dmrg_sweeps_executed"] == 0
    assert len(result["su2_state_preparation"]) == len(result["spin_projection"]) == nroots
