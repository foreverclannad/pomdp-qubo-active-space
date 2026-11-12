# PAS: active-space selection

PAS selects molecular active spaces using orbital occupations, orbital composition,
and quantum-information evidence. It accompanies *Toward Self-Driving
Multi-reference Calculations: Belief-Aware Orbital Selection*.

## Install

Use Linux or WSL with Python 3.11. The paper search replay requires a C compiler (`cc`).

```bash
python -m pip install -r requirements/replay-validated.txt
python -m pip install --no-deps .
pas --version
```

## Reproduce the paper examples

Extract all `PAS_data_*.zip` archives into one directory on a Linux filesystem.
This creates `evidence/`.

```bash
pas reproduce paper --evidence /path/to/evidence --output ./paper-results
python examples/paper/export_results.py --run ./paper-results \
    --evidence /path/to/evidence --output ./paper-tables
```

The command replays the recorded dioxetanone and TiF3 selections and analyses.
It also recomputes five small-molecule controls and the H2O symmetry comparison.
The [paper example guide](examples/paper/README.md) maps the outputs to Tables 3–7.

To run only the small molecules:

```bash
pas reproduce small-molecules --output ./small-results
```

## Analyze results from another program

```bash
pas offline example --output ./h4-example
pas offline check ./h4-example/input.json
pas offline select ./h4-example/input.json --output ./h4-result
```

The H4 example selects orbitals `[2, 3]` from supplied orbital, density and
entanglement arrays. Replace these arrays to analyze your own calculation.
See [external-data input](docs/offline-input.md). MOKIT/FCH import is optional.

Natural occupations can also be entered directly in JSON:

```bash
pas offline select ./h4-example/input-natural.json --output ./natural-result
```

The [natural-orbital guide](pas_release/offline_example/NATURAL_ORBITALS.md)
lists the accompanying data and includes an occupation-only input.

## Results and tests

`summary.txt` gives the main results, `result.json` stores full-precision values,
and `history.csv` records the selection history. Public orbital indices start
at 1; energies are in Hartree. The default composition coefficient is 0.45.

```bash
python -m pytest -q
```

[Reproduction details](docs/reproduction.md) describe the numerical checks.
Citation: `CITATION.cff`. License: [MIT](LICENSE).
