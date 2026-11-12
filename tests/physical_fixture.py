"""Synthetic solver fixtures still satisfy the real immutable request contract."""

from pathlib import Path

from pas.common import file_hash, identity, write_json


def bind_request(receipt, context, directory, config):
    kind = receipt["kind"]
    active = receipt["active"]
    partition = (
        dict(
            active=list(context.window),
            inactive=[],
            secondary=[],
            frozen=list(context.frozen),
            deleted=list(context.deleted),
            electrons=context.total_electrons - 2 * len(context.frozen),
        )
        if kind == "survey"
        else context.partition(active)
    )
    request = dict(
        schema="pas_physical_request_v1",
        context_id=context.context_id,
        basis_id=context.basis_id,
        checkpoint_hash=context.payload.get("checkpoint_hash", context.basis_id),
        kind=kind,
        partition=partition,
        protocol=config,
        input="Synthetic test fixture; no quantum-chemistry solver launched",
    )
    request["request_id"] = identity(request)
    path = Path(directory).resolve() / "request.json"
    write_json(path, request)
    receipt["request_id"] = request["request_id"]
    receipt.setdefault("artifacts", {})["request"] = dict(path=str(path), sha256=file_hash(path))
    receipt["evidence_id"] = identity({k: v for k, v in receipt.items() if k != "evidence_id"})
    return receipt
