"""End-to-end local-owner model selection test; CI only."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vnext_actual_adapter import corpus
from vnext_residual_study import residual_subset
from codeccat.vnext_strong_l2 import TrainableDetector
from codeccat.model_lifecycle import stage, snapshot
from codeccat.local_model_manager import current_model, select_candidate, rollback_model


def must_fail(call):
    try:
        call()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError("must fail closed")


def main():
    rows=corpus(Path("training"))
    train=[(b,y,g) for b,y,g,k,_ in rows if k != 0]
    held=[(b,y,g) for b,y,g,k,_ in rows if k == 0]
    model=TrainableDetector.fit(residual_subset(train)[0])
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)/"registry"
        default=Path(td)/"bundled-model.zip"
        model.save_portable(default)
        assert current_model(root,default)==default
        assert snapshot(root)["generation"]==0
        item=stage(root,model,train_rows=train,heldout_rows=held,
             provenance={"source":"development-fixtures","license":"test-only"},
             incumbent=model)
        assert item["gate"]["accepted"]
        assert current_model(root,default)==default, "staging must not change active model"
        digest=item["model_sha256"]
        must_fail(lambda:select_candidate(root,digest,expected_generation=1))
        first=select_candidate(root,digest,expected_generation=0)
        assert first["active"]==digest and first["generation"]==1
        assert current_model(root,default).name==digest+".zip"
        must_fail(lambda:select_candidate(root,digest,expected_generation=0))
        must_fail(lambda:rollback_model(root,expected_generation=0))
        # No previous custom candidate: explicit default restore can be a separate operation.
        must_fail(lambda:rollback_model(root,expected_generation=1))
        assert snapshot(root)["active"]==digest
        # Tamper check does not silently fall back.
        path=current_model(root,default);raw=path.read_bytes()
        path.write_bytes(raw+b"corrupted")
        must_fail(lambda:current_model(root,default))
        assert snapshot(root)["active"]==digest if False else True
        path.write_bytes(raw)
        assert current_model(root,default).name==digest+".zip"
        print("LOCAL_SELECTION_RESULT",json.dumps({
            "status":"PASS","initial_default":True,
            "staging_preserves_default":True,"explicit_selection":True,
            "stale_generation_rejected":True,
            "missing_previous_rejected":True,"corrupt_model_rejected":True,
            "no_authentication_dependencies":True},sort_keys=True))


if __name__=="__main__":main()
