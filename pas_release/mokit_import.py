"""Optional MOKIT FCH import, isolated from user files and native solver paths."""

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .common import require
from .resources import PACKAGE

DENSITIES = {
    "Total SCF Density": 1,
    "Total CI Density": 3,
    "Total MP2 Density": 5,
    "Total CC Density": 7,
    "Total CI Rho(1)": 9,
}


def read_orbitals(path, *, density_section=None, expected_sha256=None):
    require(
        importlib.util.find_spec("mokit") is not None
        and importlib.util.find_spec("pyscf") is not None,
        "MOKIT FCH import requires optional MOKIT and PySCF. Install with: conda install -c mokit -c conda-forge mokit=1.2.8 pyscf",
    )
    require(
        density_section is None or density_section in DENSITIES,
        "observations.fch_density: choose " + ", ".join(DENSITIES),
    )
    content = Path(path).read_bytes()
    require(
        expected_sha256 is None or hashlib.sha256(content).hexdigest() == expected_sha256,
        "Orbital file changed before MOKIT import",
    )
    # MOKIT writes intermediate files and some Fortran errors terminate Python.
    # A disposable worker keeps both behaviours outside the caller and inputs.
    with TemporaryDirectory(prefix="pas-mokit-") as temporary:
        work = Path(temporary)
        (work / "source.fch").write_bytes(content)
        command = [
            sys.executable,
            "-m",
            "pas_release.mokit_import",
            "--worker",
            "source.fch",
            "converted.npz",
        ]
        if density_section is not None:
            command += ["--density", density_section]
        env = dict(
            os.environ,
            PYTHONPATH=str(PACKAGE.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        try:
            run = subprocess.run(
                command, cwd=work, env=env, capture_output=True, text=True, timeout=90
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError("MOKIT orbital import exceeded 90 seconds") from error
        require(
            run.returncode == 0 and (work / "converted.npz").is_file(),
            "MOKIT could not import this FCH file: " + (run.stderr + run.stdout)[-1800:],
        )
        with np.load(work / "converted.npz", allow_pickle=False) as arrays:
            result = {k: arrays[k].copy() for k in arrays.files}
    version = str(result.pop("mokit_version"))
    result["electron_count"] = int(result["electron_count"])
    return result, dict(
        program="MOKIT",
        version=version,
        format="fch",
        density_section=density_section,
        orbital_rotation=False,
        quantum_chemistry_calculations=0,
    )


def read_density(path, section, orbitals):
    require(
        section in DENSITIES and "density_mo" in orbitals,
        "Requested FCH density was not imported: " + str(path),
    )
    return orbitals["density_mo"]


def _worker(path, output, section):
    import mokit
    from mokit.lib import rwwfn
    from mokit.lib.fch2py import fch2py
    from mokit.lib.gaussian import load_mol_from_fch

    # Conda-installed utilities can be absent from PATH when called via a venv.
    binary = Path(mokit.__file__).resolve().parents[4] / "bin"
    if binary.is_dir():
        os.environ["PATH"] = str(binary) + os.pathsep + os.environ.get("PATH", "")
    text = Path(path).read_text()
    require(
        "Beta MO coefficients" not in text,
        "This importer requires a common restricted spatial MO basis; unrestricted alpha/beta orbitals need explicit common-basis arrays",
    )
    if section is not None:
        require(
            any(line.startswith(section) for line in text.splitlines()),
            "FCH does not contain the requested density: " + section,
        )
    nbf, nif = rwwfn.read_nbf_and_nif_from_fch(path)
    require(
        nbf == nif and nbf > 0,
        "FCH import requires a full square MO basis; linear-dependence-trimmed bases are not supported",
    )
    mol = load_mol_from_fch(path)
    require(
        not mol.cart,
        "FCH importer currently supports real spherical Gaussian basis functions; Cartesian bases require explicit normalized arrays",
    )
    c = fch2py(path, nbf, nif, "a")
    raw_c = rwwfn.read_mo_from_fch(path, nbf, nif, "a")
    centers, angular = [], []
    for shell in range(mol.nbas):
        width = int(mol.ao_loc_nr()[shell + 1] - mol.ao_loc_nr()[shell])
        centers.extend([mol.bas_atom(shell) + 1] * width)
        angular.extend([mol.bas_angular(shell)] * width)
    arrays = dict(
        coefficients=c,
        overlap=mol.intor_symmetric("int1e_ovlp"),
        ao_centers=np.asarray(centers, int),
        ao_angular_momenta=np.asarray(angular, int),
        center_labels=np.asarray([mol.atom_symbol(i) for i in range(mol.natm)]),
        coordinates_angstrom=mol.atom_coords(unit="Angstrom"),
        electron_count=mol.nelectron,
        mokit_version=getattr(mokit, "__version__", "unknown"),
    )
    if section is not None:
        density = rwwfn.read_dm_from_fch(path, DENSITIES[section], nbf)
        # Both arrays use the original Gaussian AO convention: P = C D C^T.
        # Solve in that convention, then retain the same MO column order as fch2py.
        left = np.linalg.solve(raw_c, density)
        arrays["density_mo"] = np.linalg.solve(raw_c, left.T).T
    np.savez_compressed(output, **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal disposable MOKIT conversion worker")
    parser.add_argument("--worker", nargs=2, required=True, metavar=("INPUT", "OUTPUT"))
    parser.add_argument("--density", choices=tuple(DENSITIES))
    args = parser.parse_args()
    _worker(*args.worker, args.density)
