# Direct natural-orbital occupations

This H4/STO-3G example reads natural occupations directly from JSON and selects
CAS(2e,2o) from a four-electron, four-orbital FCI calculation. The supplied files
run with PAS, NumPy, SciPy and h5py. MOKIT and an electronic-structure solver are
optional.

## Run

After installing PAS, copy the examples into a new directory:

```bash
pas offline example --output ./h4-example
pas offline check ./h4-example/input-natural.json
pas offline select ./h4-example/input-natural.json --output ./natural-result
```

The initial orbitals are `[1, 3]`; the recommendation is `[2, 3]`. Selection
scores are 0.247377237320 and 0.319892731592. These indices refer to the columns
of `natural-orbitals.npz`. `expected-natural.json` contains full-precision
results; the comparison tolerance is 1e-10.

To use occupation evidence alone:

```bash
pas offline check ./h4-example/input-natural-occupation.json
pas offline select ./h4-example/input-natural-occupation.json --output ./occupation-result
```

This input needs only its JSON file and `natural-orbitals.npz`. Its recommendation
is also `[2, 3]`. It has no observation NPZ or QIT requirement.

## Supply your own data

Copy a complete input JSON and replace the case settings and orbital file.
Natural occupations go in `observations.natural_occupations`:

```json
"density_convention": "spin_summed_natural_orbital",
"natural_occupations": [[
  1.931013779993419,
  1.8313052314690375,
  0.17456102394480535,
  0.06311996459273583
]]
```

Keep the outer list even for one state. Rows follow `states.ids`; columns
follow `observations.orbitals`, whose indices start at 1. Values are spin-summed
spatial-orbital occupations in [0, 2]. Each row must sum to
`observations.electrons`. Use the full correlated observation space, then set
the smaller target space with `selection.electrons` and `selection.orbitals`.

| Data or setting | What to provide |
|---|---|
| Natural orbitals | A complete real MO coefficient matrix, AO overlap, AO atom/angular-momentum labels, atom labels and coordinates in `orbitals.npz`; alternatively, a supported FCH file with these natural orbitals |
| Orbital identity | `observations.orbital_file_sha256`, computed with `sha256sum` on that orbital file |
| States | State labels, normalized positive weights and spin multiplicity |
| Selection | Target electron/orbital counts, initial active indices and a reference determinant's 0/1/2 electron roles |
| Composition evidence | Target atom indices and optional angular momenta, in the same orbital/AO frame |
| QIT evidence | Single-orbital entropies and mutual information computed in these natural orbitals, stored in an observation NPZ |

Export each occupation with its corresponding natural-orbital column from the
same calculation. PAS uses this order and constructs `D[r] = diag(n[r])`.
The `spin_summed_natural_orbital` convention declares that these columns are
natural orbitals for every supplied state. Separately computed state-specific
natural orbitals cannot be combined into one input unless they share this
basis. State-averaged natural orbitals generally require full per-state density
matrices for multistate analysis.

`selection.reference_occupations` specifies a reference determinant; it is
separate from the correlated natural occupations. The supplied reference doubly
occupies natural orbitals 1 and 2. Choose these roles explicitly for your system.

Choose `occupation`, `composition` and/or `qit` under `evidence.enabled`.
Weights default to 1; composition alpha defaults to 0.45. If QIT is enabled,
provide its arrays in the same natural-orbital basis. Natural occupations do not
determine orbital entropies or mutual information. `observations.file` can be
omitted when no observation arrays are needed.

Each occupation input must use one density source: the inline natural
occupations, full density arrays, or an FCH density section. Conflicting sources,
missing enabled evidence, wrong dimensions, non-finite values and inconsistent
electron counts produce an error. The density checks allow 1e-7 roundoff at the
occupation bounds and 1e-6 in each electron sum; input values are not renormalized.

## Files and numerical check

| File | Contents |
|---|---|
| `input-natural.json` | Complete input with inline occupations and all three evidence channels |
| `input-natural-occupation.json` | Complete input using occupation evidence alone |
| `natural-orbitals.npz` | Corresponding natural orbitals, AO overlap and molecular metadata |
| `natural-qit.npz` | Entropies and mutual information in the natural-orbital basis |
| `input-natural-density.json`, `natural-observations.npz` | Full-1-RDM input for the same calculation |
| `expected-natural.json` | Expected selections, scores and candidate rankings for both occupation inputs |
| `natural-provenance.json` | Calculation method, energies, numerical checks and file hashes |

The full-density route gives the same recommendation and scores within 1e-10:

```bash
pas offline select ./h4-example/input-natural-density.json --output ./density-result
```

Both routes write `summary.txt`, `result.json`, `candidates.csv` and
`objective.npz`. Scores evaluate the supplied observations. Obtain optimized
energies for a recommended space with an electronic-structure program.

The source package includes `examples/offline/generate_natural.py`. To regenerate
the example with PySCF installed:

```bash
python examples/offline/generate_natural.py --output ./regenerated-natural
```

The script diagonalizes the FCI 1-RDM, solves FCI again in the resulting natural
orbitals and computes their orbital entropies and mutual information. It checks
the energy and AO density before exporting the inputs and expected results.
