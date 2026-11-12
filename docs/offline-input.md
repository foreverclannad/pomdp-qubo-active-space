# External-data input

PAS can select an active space from a fixed set of orbitals and observations
exported by another program. Base input requires NumPy, SciPy and h5py.

## Run the example

```bash
pas offline example --output ./h4-example
pas offline check ./h4-example/input.json
pas offline select ./h4-example/input.json --output ./h4-result
```

The supplied H4/STO-3G full-CI data select CAS(2e,2o) from four observed orbitals.
The initial space is `[1, 3]`; the recommended space is `[2, 3]`. Selection scores
are 0.247512465056 and 0.320324753146. All four legal candidates are enumerated.
`expected.json` contains the full expected result.

## Input fields

Copy the example input and replace its file paths and system settings. Paths are
relative to the input JSON. Public atom and orbital indices start at 1.

| Field | Meaning |
|---|---|
| `schema` | `pas.offline.v1` |
| `name`, `source` | Case name, generating program, method and optional version |
| `orbitals.format`, `orbitals.file` | `npz` or `mokit_fch`, and the orbital file |
| `observations.file` | NPZ file containing the selected evidence arrays |
| `observations.orbitals` | Full-MO indices defining the observation-array order |
| `observations.electrons` | Electron count in the observed space |
| `observations.orbital_file_sha256` | SHA-256 of the corresponding orbital file |
| `selection.electrons`, `selection.orbitals` | Target electron and orbital counts |
| `selection.reference_occupations` | Reference electron roles, 0/1/2, for all MOs |
| `selection.initial_active` | Initial active-space indices for comparison |
| `selection.required_active` | Optional orbitals that must be retained |
| `states` | State `ids`, normalized positive `weights` and `spin_multiplicity` |
| `evidence.enabled` | A nonempty subset of `occupation`, `composition`, `qit` |
| `evidence.weights` | Optional positive weights; default 1 for each enabled channel |
| `evidence.composition_alpha` | Composition coefficient; default 0.45 |
| `target.atoms`, `target.angular_momenta` | Composition target atoms and optional angular momenta |
| `search` | Optional enumeration and simulated-annealing settings |

Reference occupations describe electron roles, not rounded correlated occupations.
Use `sha256sum orbitals.npz` to obtain the orbital-file hash.

## Arrays and units

For N full MOs, M observed orbitals and R states, the orbital NPZ contains:

- `coefficients[N,N]`: AO rows and MO columns.
- `overlap[N,N]`: AO overlap in the same AO order; `C.T @ S @ C = I`.
- `ao_centers[N]`, `ao_angular_momenta[N]`: 1-based atom indices and angular momenta.
- `center_labels` and `coordinates_angstrom[atoms,3]`.

Supply observations for the enabled channels:

- Occupation: `densities[R,M,M]`, the full real spin-summed MO 1-RDM;
  `density_convention` is `spin_summed_mo`.
- QIT: `single_orbital_entropy[R,M]` and `mutual_information[R,M,M]`;
  `entropy_unit` is `natural_log`, and `mutual_information_convention` is
  `half_mutual_information`, meaning `0.5*(s_i+s_j-s_ij)`.
- Composition: orbital data and `target`; no extra observation array.

Use finite numeric arrays without pickle objects. The program checks symmetry,
orbital orthogonality, density traces, state dimensions and enabled evidence.

## Optional MOKIT import

```bash
conda env create -f requirements/mokit-import.yml
conda activate pas-mokit
python -m pip install .
pas offline select ./h4-example/input-mokit.json --output ./h4-mokit-result
```

The FCH example contains the same orbitals and total density, with entropy and
mutual information in a separate NPZ. FCH input supports complete, real,
restricted spatial orbitals and spherical Gaussian basis functions. Other formats
can be converted to FCH with the appropriate [MOKIT tool](https://jeanwsr.gitlab.io/mokit-doc-mdbook/chap4-5.html).
The example records the FCI density method explicitly in `source.method`.

## Output

`summary.txt` and `result.json` give the recommendation and scores.
`candidates.csv` lists candidate rankings, and `objective.npz` contains the QUBO
coefficients for `constant + linear@x + x@pair@x`. `selection_score` is the
weighted evidence score minus the QIT boundary term; `quality` is the weighted
score before that term.

These scores describe the imported wavefunction. To obtain optimized energies
for a recommended space, run that space in a quantum-chemistry program and
import the new observations. The data-generation script is
`examples/offline/generate.py`; it requires the optional PySCF/MOKIT environment.
