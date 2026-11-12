"""Physical OpenMolcas orbital I/O and metric AO projection only.

No policy, response model, window selection, optimizer or reference is imported.
Adapted physical algorithms and source hashes: docs/review/reused_sources.json.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_TYPE_INDEX_LINE = re.compile(r"^(?P<prefix>\s*\d+\s+)(?P<types>[FfIi123SsDd]+)(?P<suffix>\s*)$")


@dataclass(frozen=True)
class InpOrbData:
    """Occupation and subspace metadata read from an INPORB-format file."""

    occupations: tuple[float, ...]
    type_indices: tuple[str, ...]

    def __post_init__(self) -> None:
        occupations = tuple(float(value) for value in self.occupations)
        type_indices = tuple(str(value).lower() for value in self.type_indices)
        if not occupations or len(occupations) != len(type_indices):
            raise ValueError("INPORB occupations and type indices must have equal non-zero size")
        if (
            not np.isfinite(occupations).all()
            or min(occupations) < -1e-7
            or max(occupations) > 2 + 1e-7
        ):
            raise ValueError("INPORB occupations must be finite and inside [0,2]")
        if any(value not in {"f", "i", "1", "2", "3", "s", "d"} for value in type_indices):
            raise ValueError("INPORB contains an unsupported type index")
        object.__setattr__(self, "occupations", occupations)
        object.__setattr__(self, "type_indices", type_indices)

    @property
    def orbital_count(self) -> int:
        return len(self.occupations)

    @property
    def electron_count(self) -> int:
        total = float(sum(self.occupations))
        rounded = int(round(total))
        if not np.isclose(total, rounded, atol=1.0e-5, rtol=0.0):
            raise ValueError("INPORB occupation sum is not an integer electron count")
        return rounded


@dataclass(frozen=True)
class OpenMolcasHDF5Orbitals:
    """Full-atom AO data with MO columns in native symmetry-block order."""

    mo_coefficients: np.ndarray
    ao_overlap: np.ndarray
    energies: np.ndarray
    occupations: np.ndarray
    type_indices: tuple[str, ...]
    center_labels: tuple[str, ...]
    basis_function_ids: np.ndarray
    symmetry_block_sizes: tuple[int, ...]
    metadata: Mapping[str, Any]
    symmetry_to_ao: np.ndarray | None = None

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.mo_coefficients, dtype=float)
        overlap = np.asarray(self.ao_overlap, dtype=float)
        energies = np.asarray(self.energies, dtype=float).reshape(-1)
        occupations = np.asarray(self.occupations, dtype=float).reshape(-1)
        basis_ids = np.asarray(self.basis_function_ids, dtype=int)
        nmo = int(energies.size)
        if coefficients.shape != (nmo, nmo):
            raise ValueError("OpenMolcas MO coefficient matrix must be square")
        if overlap.shape != (nmo, nmo):
            raise ValueError("OpenMolcas AO overlap matrix must match MO size")
        if occupations.size != nmo or len(self.type_indices) != nmo:
            raise ValueError("OpenMolcas MO arrays must have equal size")
        if not all(np.isfinite(x).all() for x in (coefficients, overlap, energies, occupations)):
            raise ValueError("OpenMolcas orbital arrays must be finite")
        if not np.allclose(overlap, overlap.T, atol=1e-10, rtol=0):
            raise ValueError("OpenMolcas AO overlap is not symmetric")
        if set(str(t).strip().lower() for t in self.type_indices) - {
            "f",
            "i",
            "1",
            "2",
            "3",
            "s",
            "d",
        }:
            raise ValueError("OpenMolcas contains an unsupported orbital type")
        if basis_ids.shape != (nmo, 4):
            raise ValueError("BASIS_FUNCTION_IDS must have shape (n_ao, 4)")
        if (
            not self.symmetry_block_sizes
            or any(int(value) < 0 for value in self.symmetry_block_sizes)
            or sum(int(value) for value in self.symmetry_block_sizes) != nmo
        ):
            raise ValueError("OpenMolcas symmetry block sizes must sum to n_ao")
        if np.any(basis_ids[:, 0] < 1) or np.any(basis_ids[:, 0] > len(self.center_labels)):
            raise ValueError("basis function center IDs are outside the atom list")
        if self.symmetry_to_ao is not None:
            transform = np.asarray(self.symmetry_to_ao, dtype=float)
            if transform.shape != (nmo, nmo) or not np.all(np.isfinite(transform)):
                raise ValueError("DESYM_MATRIX must be a finite square AO transform")
            object.__setattr__(self, "symmetry_to_ao", transform)
        object.__setattr__(self, "mo_coefficients", coefficients)
        object.__setattr__(self, "ao_overlap", overlap)
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "occupations", occupations)
        object.__setattr__(self, "basis_function_ids", basis_ids)
        object.__setattr__(
            self,
            "type_indices",
            tuple(str(value).strip().lower() for value in self.type_indices),
        )
        object.__setattr__(
            self,
            "center_labels",
            tuple(str(value).strip() for value in self.center_labels),
        )
        object.__setattr__(
            self,
            "symmetry_block_sizes",
            tuple(int(value) for value in self.symmetry_block_sizes),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def orbital_count(self) -> int:
        return int(self.energies.size)

    def native_coefficients_to_ao(self, coefficients: np.ndarray) -> np.ndarray:
        """Put INPORB coefficients into the same full-atom AO basis as this object."""

        native = np.asarray(coefficients, dtype=float)
        if native.ndim != 2 or native.shape[0] != self.orbital_count:
            raise ValueError("native coefficients do not match the symmetry AO layout")
        return native if self.symmetry_to_ao is None else self.symmetry_to_ao @ native


def read_inporb(path: str | Path) -> InpOrbData:
    """Read occupations and type indices from an INPORB/RasOrb/ScfOrb file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    text = source.read_text(encoding="utf-8", errors="replace")
    occupations = _parse_numeric_section(text, "#OCC")
    type_indices = _parse_type_indices(text)
    return InpOrbData(
        occupations=tuple(occupations),
        type_indices=tuple(type_indices),
    )


def read_openmolcas_orbital_hdf5(
    path: str | Path,
) -> OpenMolcasHDF5Orbitals:
    """Read standard SCF/RASSCF HDF5 orbital data, including AO identities."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "reading OpenMolcas HDF5 files requires the optional h5py package"
        ) from exc

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with h5py.File(source, mode="r") as handle:
        block_sizes = tuple(int(value) for value in np.asarray(handle.attrs["NBAS"]).reshape(-1))
        coefficients = _square_block_matrix(
            np.asarray(handle["MO_VECTORS"][()], dtype=float),
            block_sizes,
            coefficient_blocks=True,
        )
        overlap = _square_block_matrix(
            np.asarray(handle["AO_OVERLAP_MATRIX"][()], dtype=float),
            block_sizes,
            coefficient_blocks=False,
        )
        type_indices = tuple(_decode(value) for value in handle["MO_TYPEINDICES"][()])
        labels = tuple(_decode(value) for value in handle["CENTER_LABELS"][()])
        metadata = {str(key): _json_scalar(value) for key, value in handle.attrs.items()}
        basis_ids = np.asarray(handle["BASIS_FUNCTION_IDS"][()], dtype=int)
        transform = None
        if len(block_sizes) > 1:
            required = (
                "DESYM_MATRIX",
                "DESYM_CENTER_LABELS",
                "DESYM_BASIS_FUNCTION_IDS",
                "DESYM_CENTER_COORDINATES",
            )
            missing = [name for name in required if name not in handle]
            if missing:
                raise ValueError("symmetry HDF5 lacks full-atom AO evidence: " + ", ".join(missing))
            raw_transform = np.asarray(handle["DESYM_MATRIX"][()], dtype=float)
            nbas = sum(block_sizes)
            if raw_transform.ndim != 1 or raw_transform.size != nbas * nbas:
                raise ValueError("DESYM_MATRIX must use OpenMolcas flat Fortran storage")
            # runfile_util/run2h5: fast index is the desymmetrized AO index.
            transform = raw_transform.reshape((nbas, nbas), order="F")
            if not np.all(np.isfinite(transform)):
                raise ValueError("DESYM_MATRIX contains non-finite values")
            try:
                inverse = np.linalg.solve(transform, np.eye(nbas))
            except np.linalg.LinAlgError as exc:
                raise ValueError("DESYM_MATRIX is singular") from exc
            if np.linalg.norm(transform, ord=np.inf) * np.linalg.norm(inverse, ord=np.inf) > 1.0e10:
                raise ValueError("DESYM_MATRIX is numerically singular")
            coefficients = transform @ coefficients
            overlap = inverse.T @ overlap @ inverse
            labels = tuple(_decode(value) for value in handle["DESYM_CENTER_LABELS"][()])
            basis_ids = np.asarray(handle["DESYM_BASIS_FUNCTION_IDS"][()], dtype=int)
            coordinates = np.asarray(handle["DESYM_CENTER_COORDINATES"][()], dtype=float)
            if coordinates.shape != (len(labels), 3) or not np.all(np.isfinite(coordinates)):
                raise ValueError("DESYM_CENTER_COORDINATES does not match the full atom list")
            metadata["ao_representation"] = "full_atom_desymmetrized"
            metadata["center_coordinates_bohr"] = coordinates.tolist()
        return OpenMolcasHDF5Orbitals(
            mo_coefficients=coefficients,
            ao_overlap=overlap,
            energies=np.asarray(handle["MO_ENERGIES"][()], dtype=float),
            occupations=np.asarray(handle["MO_OCCUPATIONS"][()], dtype=float),
            type_indices=type_indices,
            center_labels=labels,
            basis_function_ids=basis_ids,
            symmetry_block_sizes=block_sizes,
            metadata=metadata,
            symmetry_to_ao=transform,
        )


def select_ao_indices(
    orbitals: OpenMolcasHDF5Orbitals,
    rules: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    """Select AO rows using center/element and (n,l,m) basis-ID rules."""

    selected: set[int] = set()
    for raw_rule in rules:
        rule = dict(raw_rule)
        center_indices = None
        if "center_indices" in rule:
            raw_centers = rule["center_indices"]
            if (
                not isinstance(raw_centers, (list, tuple))
                or not raw_centers
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    or value < 0
                    or value >= len(orbitals.center_labels)
                    for value in raw_centers
                )
            ):
                raise ValueError("center_indices must be nonempty zero-based atom indices")
            center_indices = set(int(value) for value in raw_centers)
        centers = {str(value).strip() for value in rule.get("centers", ())}
        elements = {str(value).strip().casefold() for value in rule.get("elements", ())}
        radial = {int(value) for value in rule.get("radial_indices", ())}
        angular = {int(value) for value in rule.get("angular_momenta", ())}
        magnetic = {int(value) for value in rule.get("magnetic_quantum_numbers", ())}
        for ao_index, (center_id, n_value, l_value, m_value) in enumerate(
            orbitals.basis_function_ids
        ):
            label = orbitals.center_labels[int(center_id) - 1]
            element = _element_from_center_label(label)
            if center_indices is not None and int(center_id) - 1 not in center_indices:
                continue
            if centers and label not in centers:
                continue
            if elements and element.casefold() not in elements:
                continue
            if radial and int(n_value) not in radial:
                continue
            if angular and int(l_value) not in angular:
                continue
            if magnetic and int(m_value) not in magnetic:
                continue
            selected.add(int(ao_index))
    if not selected:
        raise ValueError("AO target rules selected no basis functions")
    return tuple(sorted(selected))


def ao_subspace_projection_scores(
    mo_coefficients: np.ndarray,
    ao_overlap: np.ndarray,
    ao_indices: Sequence[int],
    *,
    rcond: float = 1.0e-10,
) -> np.ndarray:
    """Return metric projector expectation values for every MO."""

    coefficients = np.asarray(mo_coefficients, dtype=float)
    overlap = np.asarray(ao_overlap, dtype=float)
    indices = np.asarray(tuple(int(index) for index in ao_indices), dtype=int)
    if coefficients.ndim != 2 or coefficients.shape[0] != coefficients.shape[1]:
        raise ValueError("MO coefficients must be a square matrix")
    if overlap.shape != coefficients.shape:
        raise ValueError("AO overlap must match the coefficient matrix")
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= overlap.shape[0]):
        raise ValueError("AO projector indices are empty or out of range")
    cross = overlap[:, indices]
    metric = overlap[np.ix_(indices, indices)]
    inverse = np.linalg.pinv(metric, rcond=float(rcond), hermitian=True)
    projected = coefficients.T @ cross
    scores = np.einsum("ik,kl,il->i", projected, inverse, projected)
    return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)


def _parse_numeric_section(text: str, section: str) -> list[float]:
    lines = str(text).splitlines()
    start = next(
        (index + 1 for index, line in enumerate(lines) if line.strip().upper() == section.upper()),
        None,
    )
    if start is None:
        raise ValueError(f"INPORB is missing {section}")
    values: list[float] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("#"):
            break
        if not stripped or stripped.startswith("*"):
            continue
        tokens = stripped.split()
        parsed_tokens: list[float] = []
        try:
            parsed_tokens = [_fortran_float(token) for token in tokens]
        except ValueError:
            parsed_tokens = []
        if parsed_tokens:
            values.extend(parsed_tokens)
            continue
        compact = line.rstrip("\r\n")
        if len(compact) % 18:
            raise ValueError(f"cannot parse numeric INPORB line: {line!r}")
        values.extend(
            _fortran_float(compact[offset : offset + 18]) for offset in range(0, len(compact), 18)
        )
    if not values:
        raise ValueError(f"INPORB {section} contains no numeric values")
    return values


def _parse_type_indices(text: str) -> list[str]:
    lines = str(text).splitlines()
    start = next(
        (index + 1 for index, line in enumerate(lines) if line.strip().upper() == "#INDEX"),
        None,
    )
    if start is None:
        raise ValueError("INPORB is missing #INDEX")
    values: list[str] = []
    for line in lines[start:]:
        if line.strip().startswith("#"):
            break
        match = _TYPE_INDEX_LINE.match(line)
        if match is not None:
            values.extend(match.group("types").lower())
    if not values:
        raise ValueError("INPORB #INDEX contains no orbital types")
    return values


def _replace_type_indices(text: str, updated: Sequence[str]) -> str:
    values = tuple(str(value).lower() for value in updated)
    lines = str(text).splitlines(keepends=True)
    inside = False
    offset = 0
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.upper() == "#INDEX":
            inside = True
            output.append(line)
            continue
        if inside and stripped.startswith("#"):
            inside = False
        if inside:
            content = line.rstrip("\r\n")
            ending = line[len(content) :]
            match = _TYPE_INDEX_LINE.match(content)
            if match is not None:
                width = len(match.group("types"))
                chunk = "".join(values[offset : offset + width])
                if len(chunk) != width:
                    raise ValueError("replacement type index is shorter than INPORB")
                output.append(match.group("prefix") + chunk + match.group("suffix") + ending)
                offset += width
                continue
        output.append(line)
    if offset != len(values):
        raise ValueError("replacement type-index size does not match the INPORB #INDEX section")
    return "".join(output)


def _square_block_matrix(
    flat: np.ndarray,
    block_sizes: Sequence[int],
    *,
    coefficient_blocks: bool,
) -> np.ndarray:
    sizes = tuple(int(value) for value in block_sizes)
    if not sizes or any(value < 0 for value in sizes) or sum(sizes) <= 0:
        raise ValueError("OpenMolcas symmetry block sizes are invalid")
    total = sum(sizes)
    output = np.zeros((total, total), dtype=float)
    flat_values = np.asarray(flat, dtype=float).reshape(-1)
    if flat_values.size != sum(value * value for value in sizes):
        raise ValueError("OpenMolcas square-block dataset has unexpected size")
    flat_offset = 0
    matrix_offset = 0
    for size in sizes:
        count = size * size
        block = flat_values[flat_offset : flat_offset + count].reshape(
            (size, size),
            order="F" if coefficient_blocks else "C",
        )
        output[
            matrix_offset : matrix_offset + size,
            matrix_offset : matrix_offset + size,
        ] = block
        flat_offset += count
        matrix_offset += size
    if flat_offset != flat_values.size:
        raise ValueError("OpenMolcas square-block dataset has unexpected size")
    return output


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _json_scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        return _decode(item) if isinstance(item, (bytes, np.bytes_)) else item
    return [
        _decode(item)
        if isinstance(item, (bytes, np.bytes_))
        else item.item()
        if hasattr(item, "item")
        else item
        for item in array.reshape(-1)
    ]


def _element_from_center_label(label: str) -> str:
    match = re.match(r"[A-Za-z]+", str(label).strip())
    if match is None:
        raise ValueError(f"cannot infer element from center label {label!r}")
    return match.group(0)


def _fortran_float(value: str) -> float:
    return float(str(value).strip().replace("D", "E").replace("d", "e"))


_ORBITAL_HEADER = re.compile(r"^\s*\*\s*ORBITAL\s+(\d+)\s+(\d+)\s*$", re.MULTILINE | re.IGNORECASE)

_FORTRAN_FLOAT = re.compile(r"[+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))[EeDd][+-]?\d+")


def _parse_rasorb_type_indices(text: str, *, expected_count: int) -> tuple[str, ...]:
    marker = re.search(r"^\s*#INDEX\s*$", text, re.MULTILINE | re.IGNORECASE)
    if marker is None:
        raise ValueError("INPORB/RasOrb has no #INDEX section")
    result: list[str] = []
    for line in text[marker.end() :].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        if stripped.startswith("#"):
            break
        match = re.match(r"^\d+\s+([FfIi123SsDd\s]+)$", stripped)
        if match is not None:
            result.extend(list("".join(match.group(1).split()).lower()))
    if len(result) != int(expected_count):
        raise ValueError(
            "INPORB/RasOrb #INDEX count differs from candidate orbital count: "
            f"{len(result)} != {int(expected_count)}"
        )
    return tuple(result)


def read_rasorb_coefficients(
    path: str | Path,
    *,
    symmetry_block_sizes: Sequence[int],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Parse ``#ORB`` coefficients and ``#INDEX`` from INPORB/RasOrb.

    The candidate HDF5 symmetry dimensions define the only permissible block
    layout.  This supports both C1 and multi-irrep files without guessing the
    basis shape from coefficient text.
    """

    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8", errors="replace")
    orb_marker = re.search(r"^\s*#ORB\s*$", text, re.MULTILINE | re.IGNORECASE)
    if orb_marker is None:
        raise ValueError("INPORB/RasOrb has no #ORB section")
    section_tail = text[orb_marker.end() :]
    next_marker = re.search(r"^\s*#[A-Za-z]", section_tail, re.MULTILINE)
    section_end = len(text) if next_marker is None else orb_marker.end() + next_marker.start()
    headers = [match for match in _ORBITAL_HEADER.finditer(text, orb_marker.end(), section_end)]
    if not headers:
        raise ValueError("INPORB/RasOrb #ORB section is empty")
    sizes = tuple(int(value) for value in symmetry_block_sizes)
    if not sizes or any(value < 0 for value in sizes) or sum(sizes) <= 0:
        raise ValueError("candidate symmetry block sizes are invalid")
    offsets = np.cumsum((0, *sizes))
    total = int(sum(sizes))
    coefficients = np.zeros((total, total), dtype=float)
    observed: set[tuple[int, int]] = set()
    for position, header in enumerate(headers):
        symmetry = int(header.group(1))
        orbital = int(header.group(2))
        key = (symmetry, orbital)
        if key in observed:
            raise ValueError(f"duplicate INPORB orbital block {key}")
        observed.add(key)
        if symmetry < 1 or symmetry > len(sizes):
            raise ValueError(f"INPORB orbital symmetry {symmetry} is outside candidate layout")
        size = sizes[symmetry - 1]
        if orbital < 1 or orbital > size:
            raise ValueError(
                f"INPORB orbital number {orbital} is outside symmetry block {symmetry}"
            )
        start = header.end()
        end = headers[position + 1].start() if position + 1 < len(headers) else section_end
        values = np.asarray(_parse_fortran_coefficient_lines(text[start:end]), dtype=float)
        if values.size != size:
            raise ValueError(
                f"INPORB orbital {key} has {values.size} coefficients, expected {size}"
            )
        row_start = int(offsets[symmetry - 1])
        column_start = row_start
        coefficients[row_start : row_start + size, column_start + orbital - 1] = values
    expected = {
        (symmetry, orbital)
        for symmetry, size in enumerate(sizes, start=1)
        for orbital in range(1, size + 1)
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"INPORB orbital blocks do not match candidate layout; missing={missing}, extra={extra}"
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("INPORB coefficients contain non-finite values")
    return coefficients, _parse_rasorb_type_indices(text, expected_count=total)


def _parse_fortran_coefficient_lines(block: str) -> list[float]:
    """Parse both whitespace-separated v2 and fixed-width v1 INPORB rows.

    Old OpenMolcas files concatenate positive fields.  A generic floating
    regex then cannot distinguish ``E+01`` followed by a value beginning in
    ``0`` from an exponent of ``+010``.  Those rows use strict 18-character
    fields, so fixed-width decoding must take precedence.
    """

    values: list[float] = []
    for raw_line in block.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("*") or stripped.startswith("#"):
            continue
        whitespace_tokens = stripped.split()
        tokens: list[str] | None = None
        if len(whitespace_tokens) > 1 and all(
            _FORTRAN_FLOAT.fullmatch(token) for token in whitespace_tokens
        ):
            tokens = whitespace_tokens
        elif len(whitespace_tokens) == 1 and _FORTRAN_FLOAT.fullmatch(stripped):
            tokens = [stripped]
        else:
            # INPORB 1.1 writes four adjacent 18-character E-format fields.
            for width in (18, 20, 22, 24):
                if len(stripped) % width:
                    continue
                chunks = [
                    stripped[offset : offset + width].strip()
                    for offset in range(0, len(stripped), width)
                ]
                if chunks and all(_FORTRAN_FLOAT.fullmatch(chunk) for chunk in chunks):
                    tokens = chunks
                    break
        if tokens is None:
            raise ValueError(f"cannot parse INPORB coefficient row: {stripped[:80]!r}")
        values.extend(float(token.replace("D", "E").replace("d", "e")) for token in tokens)
    return values
