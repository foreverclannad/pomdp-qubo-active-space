# PAS reviewer example: external H4 observations

This directory contains exported H4/STO-3G full-CI data, two equivalent input
routes, and reference results. All orbital indices are one-based.

## Run without MOKIT or a quantum-chemistry solver

After installing PAS, run from this directory:

```bash
pas offline check input.json
pas offline select input.json --output result-arrays
```

The recommended CAS(2e,2o) orbitals are `[2, 3]`, compared with the supplied
initial choice `[1, 3]`. The selection score changes from approximately
`0.247512465056` to `0.320324753146`. Four legal candidates are enumerated.
See `expected.json` for the full-precision reference.

`result-arrays/summary.txt` gives the readable result; `result.json` and
`candidates.csv` contain the evidence components and QIT boundary penalty.
`objective.npz` exports the actual QUBO. Choose a new output directory for each
run; existing files are not overwritten.

## Try the optional MOKIT route

Install MOKIT 1.2.8 and PySCF from the official MOKIT/conda-forge channels
(the source package includes `requirements/mokit-import.yml`), then run:

```bash
pas offline check input-mokit.json
pas offline select input-mokit.json --output result-mokit
```

This reads the same orbital basis and correlated density from `orbitals.fch`.
The entropy and mutual information remain in `qit.npz`. The recommended
orbitals agree; numerical scores are compared at `1e-7`, accounting for FCH
text precision. No SCF or DMRG is launched during either import route.

The FCH `Total SCF Density` section intentionally contains the exported **FCI
total density**, as declared in the input's `source.method`. Read the specified
density section; do not infer the method from its label or interpret orbital
energies as natural occupations.

## Use your own data

Edit a copy of `input.json`. Supply the orbital arrays, complete per-state
spin-summed MO densities, and the entropy/MI arrays for enabled evidence.
Update the orbital-file SHA-256 binding. State rows and measured orbital columns
must follow `states.ids` and `observations.orbitals` exactly. Target AO atoms,
active electron/orbital counts, the initial comparison space and reference
0/1/2 electron roles are explicit inputs. Evidence weights default to one,
composition alpha to 0.45. Missing enabled evidence is an error.

The NPZ route requires no MOKIT or PySCF. The FCH importer accepts a full real
restricted spatial MO basis with spherical Gaussian functions. It preserves
MO columns and converts AO conventions; it does not localize or rotate MOs.

The reported scores describe fixed imported observations. They are not newly
optimized CAS energies, and this command does not execute the full POMDP
solver-dispatch loop. Compute a proposed space with your own electronic-structure
program to assess its resulting energy.

The full schema and validation rules are in `docs/offline-input.md` in
the source package. `examples/offline/generate.py` regenerates the physical
data using optional PySCF and MOKIT; unlike the commands above, that script
explicitly performs new RHF/FCI calculations. `provenance.json` records data
origins. The paper examples are described in examples/paper/README.md.
