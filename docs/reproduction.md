# Reproduction

`pas reproduce paper` replays the recorded orbital choices, reconstructs
scores and energies from calculation outputs, compares reference spaces,
and recomputes the small-molecule controls.

The checks cover five main trajectories, 31 events, 19 optimization choices,
24 physical endpoints, the A/B/C decision comparison and the CI-CC control.
CI-OO uses the paper's coordinates throughout its inputs, outputs and reference.
`pas verify --evidence /path/to/evidence` checks the source and data hashes.

The paper configuration uses composition alpha = 0.45 and unit evidence weights.
The recorded initial dio DMRG calculations used alpha = 0.5. Replay evaluates
both coefficients and verifies the same selected orbitals and acceptance decisions.
`validation/scoring_changes.csv` gives the score differences. The energies,
densities and reference-space comparisons use the same calculation data.

The default tolerance for replayed values is 1e-12. Orbital sets and acceptance
decisions must match exactly. Small-molecule energy differences must be below
1e-8 Hartree and projection differences below 1e-7; selected orbitals and occupation
classes must match. The minimum H2O C1/C2v overlap singular value must be at least
0.999999.

Dioxetanone and TiF3 use archived quantum-chemistry outputs. The five small
molecules and the additional H2O symmetry calculation run fresh SCF calculations.
The validated Python environment is listed in `requirements/replay-validated.txt`.

The data contain calculation inputs, orbital files, density and entanglement
arrays, requests, receipts and reference subspaces. Numerical calculation outputs
retain their recorded contents; distribution manifests organize the supplied
examples. File hashes are recorded in the data and source manifests.
