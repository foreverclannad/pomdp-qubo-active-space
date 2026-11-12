"""Use the energy of the same final density supplied to Molcas's gradient.

Block2's sweep energy can precede the last truncation/rotation. At finite
Davidson tolerances it can differ from the final MPS density by microhartrees.
The pinned Molcas interface exports X[p,q,r,s] = Gamma[p,q,r,s]/2, where
Gamma = sum_spin <a+_p a+_q a_r a_s>, in the original FCIDUMP orbital order.
This correction is bookkeeping, not tighter optimization or a finite-M bound.
"""

import os
from pathlib import Path

import numpy as np

from pas.backend import fcidump
from pas.common import file_hash, read_json, write_json


def integrals(path):
    header, records = fcidump(path)
    n = header["NORB"]
    if not 1 <= n <= 64 or header["MS2"] != 0 or not 2 <= header["NELEC"] <= 2 * n:
        raise ValueError("Expected a small spin-restricted singlet FCIDUMP")
    h = np.zeros((n, n))
    v = np.zeros((n,) * 4)
    for (i, j, k, ell), value in records.items():
        if (i, j, k, ell) == (0, 0, 0, 0):
            continue
        if k == ell == 0 and 1 <= min(i, j) <= max(i, j) <= n:
            h[i - 1, j - 1] = h[j - 1, i - 1] = value
        elif 1 <= min(i, j, k, ell) <= max(i, j, k, ell) <= n:
            i, j, k, ell = i - 1, j - 1, k - 1, ell - 1
            for ix in (
                (i, j, k, ell),
                (j, i, k, ell),
                (i, j, ell, k),
                (j, i, ell, k),
                (k, ell, i, j),
                (ell, k, i, j),
                (k, ell, j, i),
                (ell, k, j, i),
            ):
                if v[ix] and abs(v[ix] - value) > 1e-12:
                    raise ValueError("Conflicting ERI symmetry")
                v[ix] = value
        else:
            raise ValueError("Unsupported integral index convention")
    return header, h, v, records[(0, 0, 0, 0)]


def density(path, n):
    with Path(path).open() as f:
        if int(f.readline()) != n:
            raise ValueError("Density orbital count differs from FCIDUMP")
        rows = np.loadtxt(f, ndmin=2)
    if rows.shape != (n**4, 5) or not np.all(np.isfinite(rows)):
        raise ValueError("Incomplete or nonfinite Molcas spatial 2-RDM")
    ix = rows[:, :4].astype(int)
    if not np.array_equal(ix, rows[:, :4]) or np.any(ix < 0) or np.any(ix >= n):
        raise ValueError("Invalid density indices")
    flat = np.ravel_multi_index(ix.T, (n,) * 4)
    if not np.array_equal(np.sort(flat), np.arange(n**4)):
        raise ValueError("Duplicate or missing density indices")
    result = np.empty(n**4)
    result[flat] = rows[:, 4]
    return result.reshape((n,) * 4)


def density_energy(h, eri, core, x, nelec):
    n = len(h)
    x = np.asarray(x, dtype=float)
    if x.shape != (n,) * 4 or not np.all(np.isfinite(x)):
        raise ValueError("Bad density shape or values")
    if not np.allclose(x, x.transpose(3, 2, 1, 0), atol=1e-7, rtol=0):
        raise ValueError("Non-Hermitian spatial 2-RDM")
    one = 2 * np.einsum("pqqs->ps", x) / (nelec - 1)
    trace = float(np.trace(one))
    if abs(trace - nelec) > 1e-6 or not np.allclose(one, one.T, atol=1e-7, rtol=0):
        raise ValueError("Density contraction does not match the declared electrons")
    energy = float(core + np.einsum("pq,pq", h, one) + np.einsum("psqr,pqrs", eri, x))
    if not np.isfinite(energy):
        raise ValueError("Nonfinite density energy")
    return dict(energy=energy, electron_trace=trace)


def reconcile(work, audit, nroots):
    """Called only after Block2 exits successfully, before Molcas reads outputs."""
    work, audit = Path(work), Path(audit)
    if type(nroots) is not int or nroots < 1:
        raise ValueError("The Molcas interface requires a positive integer root count")
    target = work / "block.energy"
    receipt = audit / "energy_density.json"
    if receipt.exists():
        record = read_json(receipt)
        if len(record["states"]) != nroots or len(record["original_sweep_energies"]) != nroots:
            raise ValueError("Recorded density-energy root count differs")
        if (
            any(file_hash(p) != sha for p, sha in record["source_hashes"].items())
            or file_hash(target) != record["effective_sha256"]
        ):
            raise ValueError("Recorded density-energy interface changed")
        return record
    original = audit / "block.energy.sweep"
    if original.exists():
        raise ValueError("Incomplete energy-interface transaction requires reconciliation")
    inputs = [
        work / "FCIDUMP",
        *[work / "node0" / f"spatial_twopdm.{i}.{i}.txt" for i in range(nroots)],
    ]
    hashes = {str(p): file_hash(p) for p in inputs}
    header, h, v, core = integrals(inputs[0])
    states = [
        density_energy(h, v, core, density(p, header["NORB"]), header["NELEC"]) for p in inputs[1:]
    ]
    raw = target.read_bytes()
    reported = np.loadtxt(target, ndmin=1)
    if reported.shape != (nroots,) or not np.all(np.isfinite(reported)):
        raise ValueError("Malformed original Block2 state energies")
    if any(file_hash(p) != sha for p, sha in hashes.items()):
        raise ValueError("Physical inputs changed during energy reconciliation")
    energies = np.array([s["energy"] for s in states])
    gap = energies - reported
    # Preserve original sweep values; never alter an archived historical result.
    original.write_bytes(raw)
    temporary = target.with_name("block.energy.density.tmp")
    temporary.write_text("".join("%25.12f\n" % e for e in energies))
    os.replace(temporary, target)
    record = dict(
        schema="pas_molcas_density_energy_v2",
        source_hashes=hashes,
        original_sha256=file_hash(original),
        effective_sha256=file_hash(target),
        states=states,
        original_sweep_energies=reported.tolist(),
        energy_density_gaps=gap.tolist(),
        maximum_energy_density_gap=float(np.max(abs(gap))),
        density_convention="X_pqrs=0.5*sum_spin<a+_p a+_q a_r a_s>",
        orbital_order="original_FCIDUMP",
        energy_source="final_density_supplied_to_Molcas",
        solver_thresholds_changed=False,
        additional_physical_executions=0,
        finite_m_error_certified=False,
    )
    write_json(receipt, record)
    print("[PAS_MOLCAS_DENSITY_ENERGY]", record["maximum_energy_density_gap"], flush=True)
    return record
