from pathlib import Path
import tempfile
import pytest
from codeccat.vnext_strong_l2 import TrainableDetector,SCHEMA

def training_data():
    return [
        (("日本語の説明と情報"*40).encode("shift_jis"),"shift_jis","j1"),
        (("日本の経済について"*40).encode("shift_jis"),"shift_jis","j2"),
        (("English modern language "*40).encode("utf-8"),"utf-8","e1"),
        (("Modern writing sample "*40).encode("utf-8"),"utf-8","e2"),
    ]

def test_train_detect_save_reload_trusted():
    model=TrainableDetector.fit(training_data())
    assert len(model.fingerprint)==64
    for raw,_,_ in training_data():
        result=model.rank(raw)
        assert result.candidates
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/"l2.joblib"
            model.save(f)
            loaded=TrainableDetector.load_trusted(f)
            assert loaded.rank(raw)==result

def test_reject_conflicting_labels():
    with pytest.raises(ValueError,match="conflicting"):
        TrainableDetector.fit([(b"same","utf-8","a"),(b"same","latin1","b")])

def test_eager_bom_does_not_mislabel_untrained_utf32():
    model=TrainableDetector.fit(training_data())
    result=model.rank("Hello".encode("utf-32"))
    assert result.encoding is None
    assert result.ambiguous

def test_utf8_bom_fastpath_and_unsupported_utf32_abstain():
    data=training_data()+[(("UTF8 BOM text"*20).encode("utf-8-sig"),"utf-8-sig","bom1")]
    model=TrainableDetector.fit(data)
    r=model.rank("A UTF8 BOM document".encode("utf-8-sig"))
    assert r.encoding=="utf-8-sig"
    assert r.bom_hint=="utf-8-sig"
    assert len(r.candidates)==1
    unknown=model.rank("A UTF32 document".encode("utf-32"))
    assert unknown.encoding is None
    assert unknown.ambiguous
