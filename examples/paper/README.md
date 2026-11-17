# Paper examples

All commands run locally on Linux or WSL. Install the dependencies listed in
the root README and extract the data archives from the
[PAS 1.0.0 release](https://github.com/foreverclannad/pomdp-qubo-active-space/releases/tag/v1.0.0)
before running the paper replay.

## Small molecules

```bash
pas reproduce small-molecules --output ./small-results
```

This runs five SCF calculations and a second H2O calculation with C2v symmetry.
The molecular inputs are in `pas_release/data/small_molecules/inputs/`.

| System | Active space | Projection sum |
|---|---|---:|
| H2 | (2e,2o) | 2.0000 |
| HF | (8e,5o) | 4.9949 |
| H2O | (8e,6o) | 5.9887 |
| NH3 | (8e,7o) | 6.9775 |
| O2, pi/pi* | (6e,4o) | 4.0000 |

These are the Table 3 results. The six H2O C1/C2v subspace singular values
round to 1.0000.

## Dioxetanone and TiF3

```bash
pas reproduce paper --evidence /path/to/evidence --output ./paper-results
python examples/paper/export_results.py --run ./paper-results \
    --evidence /path/to/evidence --output ./paper-tables
```

| Example | Optimizations | Initial → final reference coverage |
|---|---:|---|
| dio S1-min, DMRG | 2 | 8/13 → 13/13 |
| dio CI-OO, DMRG | 1 | 11/13 → 13/13 |
| dio S1-min, CASSCF | 4 | 8/13 → 13/13 |
| dio CI-OO, CASSCF | 6 | 11/13 → 12/13 |

The CI-OO input, trajectory and reference use the paper's CI-OO geometry.
The CI-CC control retains 13/13 reference directions. These values reproduce
Tables 5 and 6. S1-min is named `NX1` in calculation records.

TiF3 uses six candidate optimizations. The initial pair-averaged excitation
energies are 5254.7360 and 22060.7250 cm-1; the final values are 5733.0881 and
22330.7881 cm-1 (Table 4). A pair average uses roots 2/3 or roots 4/5 relative
to root 1.

The S1-min fourth-step A/B/C comparison gives coverage 13/12/12. B and C lie
4.503 mEh above A (Table 7). The replay also checks the prediction-error analysis.

## Exported files

- `active_spaces.csv`: final active orbitals, scores and average energies.
- `root_energies.csv`: initial/final state energies.
- `excitation_energies.csv`: dio gaps and TiF3 pair-averaged excitations.
- `reference_subspaces.csv`: initial/final reference coverage.
- `decision_control.csv`: the A/B/C comparison.
- `small_molecules.csv`: the five valence-space controls.
- `orbital_occupations.csv`: diagonal density elements in the associated MO basis.
- `molecular_inputs/`: geometries, basis specifications and state settings.

Orbital indices refer to the basis identified by `basis_sha256`. Diagonal
density elements are natural occupations only when that basis diagonalizes
the density. `controls.json` and `water_symmetry.json` provide further detail.

The H4 external-data example is described in `docs/offline-input.md`.
