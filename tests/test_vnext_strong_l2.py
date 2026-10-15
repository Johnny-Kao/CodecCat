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

def test_sparse_linear_score_equivalence_to_sklearn():
    import numpy as np
    model=TrainableDetector.fit(training_data())
    for data,_,_ in training_data():
        x=model.vectorizer.transform([data[:4096].decode("latin1")])
        row=x.getrow(0)
        direct=model.classifier.coef_[:,row.indices] @ row.data + model.classifier.intercept_
        sklearn=model.classifier.decision_function(x)
        np.testing.assert_allclose(np.ravel(direct),np.ravel(sklearn),rtol=1e-11,atol=1e-11)

def test_multiclass_rank_agrees_with_sklearn_raw_scores():
    import numpy as np
    data=training_data()+[
        (("Русский язык информация "*30).encode("cp1251"),"cp1251","ru1"),
        (("Другие русские слова "*30).encode("cp1251"),"cp1251","ru2")
    ]
    model=TrainableDetector.fit(data)
    assert len(model.classifier.classes_) == 3
    for raw,_,_ in data:
        x=model.vectorizer.transform([raw.decode("latin1")])
        expected=model.classifier.classes_[np.argsort(model.classifier.decision_function(x)[0])[::-1]]
        got=model.rank(raw)
        expected_viable=[]
        import codecs
        for label in expected:
            try:raw.decode(str(label),"strict");expected_viable.append(str(label))
            except (UnicodeError,LookupError):pass
        assert got.encoding==expected_viable[0]
