"""Prototype authenticated approval and serialized transitions for local registry.

Research-only. An HMAC attests a trusted issuer holding the secret, not a real
human identity. Production must inject approval via a trusted external channel.
"""
import hashlib,hmac,json,re
from contextlib import contextmanager
from pathlib import Path
from .model_lifecycle import _canonical,_atomic_json,_load_state,_verify_model,SCHEMA

@contextmanager
def _lock(root):
    import fcntl
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/".state.lock").open("a+b") as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def approval_token(secret,*,action,digest,expected_generation,actor):
    if action not in ("activate","rollback") or len(secret)<32 or not actor or re.fullmatch(r"[a-f0-9]{64}",digest) is None or type(expected_generation) is not int or expected_generation<0:
        raise ValueError("invalid approval request")
    payload={"schema":"codeccat-approval-v1","action":action,"digest":digest,
             "expected_generation":expected_generation,"actor":actor}
    sig=hmac.new(secret,_canonical(payload),hashlib.sha256).hexdigest()
    return {"payload":payload,"signature":sig}

def _authenticate(token,secret,action,digest,generation):
    if not isinstance(token,dict) or len(secret)<32:raise PermissionError("approval token required")
    payload=token.get("payload")
    if not isinstance(payload,dict) or payload.get("action")!=action or payload.get("digest")!=digest or payload.get("expected_generation")!=generation:
        raise PermissionError("approval not valid for this transition")
    if set(payload)!={"schema","action","digest","expected_generation","actor"} or not payload.get("actor") or payload.get("schema")!="codeccat-approval-v1":
        raise PermissionError("invalid approval issuer")
    expected=hmac.new(secret,_canonical(payload),hashlib.sha256).hexdigest()
    if not isinstance(token.get("signature"),str) or not hmac.compare_digest(expected,token["signature"]):
        raise PermissionError("invalid approval signature")
    return payload

def sign_certification(secret, *, digest, confirmation_sha256, candidate_sha256, reviewer):
    """Trusted reviewer signs immutable evidence bytes; tests may use a fixture issuer."""
    if len(secret)<32 or not reviewer or any(re.fullmatch(r"[a-f0-9]{64}",x) is None for x in (digest,confirmation_sha256,candidate_sha256)):
        raise ValueError("invalid certification request")
    payload={"schema":"codeccat-certification-v1","digest":digest,
             "confirmation_sha256":confirmation_sha256,
             "candidate_sha256":candidate_sha256,"reviewer":reviewer}
    return {"payload":payload,"signature":hmac.new(secret,_canonical(payload),hashlib.sha256).hexdigest()}

def _verify_certification(root,digest,secret):
    file=Path(root)/"candidates"/(digest+".certification.json")
    try:
        certificate=json.loads(file.read_text())
        payload=certificate["payload"]
        if payload.get("schema")!="codeccat-certification-v1" or payload.get("digest")!=digest or not payload.get("reviewer"):
            raise PermissionError("certification mismatch")
        if payload.get("candidate_sha256")!=hashlib.sha256((Path(root)/"candidates"/(digest+".json")).read_bytes()).hexdigest():
            raise PermissionError("candidate evidence changed")
        if payload.get("confirmation_sha256")!=hashlib.sha256((Path(root)/"candidates"/(digest+".confirmation.json")).read_bytes()).hexdigest():
            raise PermissionError("confirmation evidence changed")
        expected=hmac.new(secret,_canonical(payload),hashlib.sha256).hexdigest()
        if not isinstance(certificate.get("signature"),str) or not hmac.compare_digest(expected,certificate["signature"]):
            raise PermissionError("certification signature invalid")
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
        raise PermissionError("missing or invalid signed certification") from exc

def transition(root,*,action,digest,token,secret,reviewer_secret=None):
    """Local filesystem CAS + exclusive advisory lock, with replay protection."""
    if action not in ("activate","rollback") or re.fullmatch(r"[a-f0-9]{64}",digest) is None:
        raise ValueError("invalid transition request")
    root=Path(root)
    with _lock(root):
        state=_load_state(root)
        target=digest if action=="activate" else state["previous"]
        if target is None or target!=digest:
            raise ValueError("rollback target unavailable")
        payload=_authenticate(token,secret,action,digest,state["generation"])
        if action=="activate":
            record=json.loads((root/"candidates"/(digest+".json")).read_text())
            confirmation=json.loads((root/"candidates"/(digest+".confirmation.json")).read_text())
            if not (record.get("model_sha256")==digest and record.get("promotable") is True
                    and record.get("gate",{}).get("accepted") is True
                    and confirmation.get("digest")==digest
                    and confirmation.get("gate_accepted") is True
                    and confirmation.get("independence_certified") is True
                    and confirmation.get("provenance_reviewed") is True):
                raise PermissionError("candidate not certified")
        if action=="activate":
            if reviewer_secret is None or len(reviewer_secret)<32:
                raise PermissionError("trusted independent reviewer secret required")
            _verify_certification(root,digest,reviewer_secret)
        _verify_model(root,digest)
        if state["active"] is not None:_verify_model(root,state["active"])
        if digest==state["active"]:raise ValueError("already active")
        next_state={"schema":SCHEMA,"active":digest,"previous":state["active"],
                    "generation":state["generation"]+1}
        _atomic_json(root/"active.json",next_state)
        return next_state
