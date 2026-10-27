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

def test_bom_generic_model_label_preferred_to_endian_subclass():
    data=training_data()+[
        ("UTF16 text".encode("utf-16"),"utf-16","u16"),
        ("UTF16LE text".encode("utf-16-le"),"utf-16-le","u16le"),
        ("UTF32 text".encode("utf-32"),"utf-32","u32"),
        ("UTF32LE text".encode("utf-32-le"),"utf-32-le","u32le"),
    ]
    model=TrainableDetector.fit(data)
    assert model.rank("Hi".encode("utf-16")).encoding=="utf-16"
    assert model.rank("Hi".encode("utf-32")).encoding=="utf-32"

def test_bomless_utf16_ascii_null_position_evidence():
    training=training_data()+[
      (("ASCII-heavy meaningful text "*20).encode("utf-16-le"),"utf-16-le","le1"),
      (("Other ASCII-heavy prose "*20).encode("utf-16-be"),"utf-16-be","be1"),
    ]
    model=TrainableDetector.fit(training)
    for label in ("utf-16-le","utf-16-be"):
        raw=("This has many ASCII alphabetic characters "*12).encode(label)
        assert model.rank(raw).encoding==label

def test_portable_numeric_only_model_roundtrip():
    model=TrainableDetector.fit(training_data())
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/"model.ccnext"
        model.save_portable(path)
        restored=TrainableDetector.load_portable(path)
        assert restored.fingerprint==model.fingerprint
        for raw,_,_ in training_data():
            before=model.rank(raw)
            after=restored.rank(raw)
            assert before.encoding==after.encoding
            assert before.candidates==after.candidates
            import numpy as np
            np.testing.assert_allclose(before.scores,after.scores,rtol=1e-11,atol=1e-11)

def test_portable_bundle_rejects_unknown_schema():
    import json,zipfile
    model=TrainableDetector.fit(training_data())
    with tempfile.TemporaryDirectory() as folder:
        f=Path(folder)/"bundle.ccnext"
        model.save_portable(f)
        with zipfile.ZipFile(f,"r") as archive:
            manifest=json.loads(archive.read("manifest.json"))
            weights=archive.read("weights.npz")
        manifest["schema"]="incompatible"
        with zipfile.ZipFile(f,"w") as archive:
            archive.writestr("manifest.json",json.dumps(manifest))
            archive.writestr("weights.npz",weights)
        with pytest.raises(ValueError,match="schema"):
            TrainableDetector.load_portable(f)

def test_portable_multiclass_model_roundtrip():
    import numpy as np
    samples=training_data()+[
        (("Русский текст на русском языке "*24).encode("cp1251"),"cp1251","ru1"),
        (("Информация на русском языке "*24).encode("cp1251"),"cp1251","ru2"),
    ]
    detector=TrainableDetector.fit(samples)
    assert len(detector.classifier.classes_)==3
    with tempfile.TemporaryDirectory() as folder:
        archive=Path(folder)/"multiclass.ccnext"
        detector.save_portable(archive)
        loaded=TrainableDetector.load_portable(archive)
        for raw,_,_ in samples:
            original=detector.rank(raw)
            repeated=loaded.rank(raw)
            assert original.encoding==repeated.encoding
            assert original.candidates==repeated.candidates
            np.testing.assert_allclose(original.scores,repeated.scores,rtol=1e-11,atol=1e-11)

def test_utf8_bom_4096_byte_incomplete_prefix_is_not_rejected():
    data=training_data()+[
        (("日本語"*700).encode("utf-8-sig"),"utf-8-sig","sig-prefix")
    ]
    model=TrainableDetector.fit(data)
    sample=("日本語"*800).encode("utf-8-sig")[:4096]
    assert len(sample)==4096
    expected=model.rank(sample,final=False)
    assert expected.encoding=="utf-8-sig"
    assert not expected.ambiguous

def test_iso2022_escape_gate_with_supported_labels():
    sample=training_data()+[
        (("日本語の文章"*40).encode("iso2022_jp"),"iso2022_jp","jp-esc"),
        (("한국어 텍스트"*40).encode("iso2022_kr"),"iso2022_kr","kr-esc"),
    ]
    m=TrainableDetector.fit(sample)
    for encoding,text in (("iso2022_jp","日本語の文章"*20),
                          ("iso2022_kr","한국어 텍스트"*20)):
        r=m.rank(text.encode(encoding))
        assert r.encoding==encoding
        assert r.candidates==(encoding,)

def test_escape_gate_never_forces_untrained_codec():
    m=TrainableDetector.fit(training_data())
    r=m.rank(("日本語"*20).encode("iso2022_jp"))
    assert r.encoding!="iso2022_jp"

def test_terminal_escape_without_iso2022_designation_is_not_forced():
    from codeccat.vnext_escape import select_escape_encoding
    choices=("iso2022_jp","iso2022_kr")
    assert select_escape_encoding(b"\x1b[31mRED\x1b[0m",choices,final=True) is None
    assert select_escape_encoding(b"\x1b$",choices,final=False) is None

def test_bomless_utf32_ascii_lane_evidence():
    data=training_data()+[
      (("A longer ASCII paragraph for testing. "*15).encode("utf-32-le"),"utf-32-le","u32le"),
      (("Another ASCII text body for evidence "*15).encode("utf-32-be"),"utf-32-be","u32be"),
    ]
    model=TrainableDetector.fit(data)
    for label in ("utf-32-le","utf-32-be"):
        raw=("English alphabetic text. "*20).encode(label)
        got=model.rank(raw)
        assert got.encoding==label
        assert got.candidates==(label,)
def test_bomless_utf32_not_forced_on_utf16_or_untrained_class():
    from codeccat.vnext_unicode_structure import select_utf32_no_bom
    labels=("utf-32-le","utf-32-be")
    assert select_utf32_no_bom(("English text "*30).encode("utf-16-le"),labels,final=True) is None
    assert select_utf32_no_bom(("English text "*30).encode("utf-16-be"),labels,final=True) is None
    assert select_utf32_no_bom(("English text "*30).encode("utf-32-le"),(),final=True) is None

def test_opt_in_13_profile_portable_roundtrip(tmp_path):
    samples=training_data()
    samples += [
        (("More mixed language examples "+str(i)).encode("utf-8"),"utf-8","new-"+str(i))
        for i in range(4)
    ]
    model=TrainableDetector.fit(samples,feature_profile="experimental13")
    assert model.vectorizer.ngram_range == (1,3)
    name=tmp_path/"experimental13.zip"
    model.save_portable(name)
    loaded=TrainableDetector.load_portable(name)
    assert loaded.vectorizer.ngram_range==(1,3)
    for raw,y,g in samples:
        assert loaded.rank(raw).encoding==model.rank(raw).encoding

def test_unknown_profile_rejected():
    import pytest
    with pytest.raises(ValueError,match="unknown feature profile"):
        TrainableDetector.fit(training_data(),feature_profile="invalid")


def test_opt_in_structure_first_keeps_model_ranking_and_truth():
    data = training_data() + [
        (("Русский текст и пояснение " * 20).encode("cp1251"), "cp1251", "ru"),
        (("Latin accented café déjà vu " * 20).encode("cp1252"), "cp1252", "latin"),
    ]
    model = TrainableDetector.fit(data, feature_profile="experimental13")
    for raw, truth, _group in data:
        evidence = model.analyze_structure(raw)
        assert truth in evidence.possible
        assert truth not in evidence.impossible
        old = model.rank(raw)
        integrated = model.rank(raw, structural_first=True)
        assert integrated.encoding == old.encoding
        assert integrated.candidates == old.candidates

def test_structure_first_preserves_partial_multibyte_and_ambiguous_inputs():
    model = TrainableDetector.fit(training_data(), feature_profile="experimental13")
    prefixes = [
        (b"\\xe3\\x81", False),
        ((("日本語" * 700).encode("utf-8"))[:4096], False),
        (b"ASCII input is genuinely ambiguous", True),
        ("日本語の文章" .encode("shift_jis"), True),
    ]
    for raw, final in prefixes:
        evidence = model.analyze_structure(raw, final=final)
        assert "utf-8" in evidence.possible or final
        old = model.rank(raw, final=final)
        integrated = model.rank(raw, final=final, structural_first=True)
        assert integrated.encoding == old.encoding
        assert integrated.candidates == old.candidates

def test_structural_evidence_excludes_proven_impossible_without_forcing_a_family():
    model = TrainableDetector.fit(training_data())
    evidence = model.analyze_structure(b"\\xff\\xfeInvalid UTF-8 payload")
    assert "utf-8" in evidence.impossible
    assert "shift_jis" in evidence.possible or "shift_jis" in evidence.impossible
    # A strict-decoder gate is an impossibility test, not a unique label oracle.
    assert len(evidence.possible) + len(evidence.impossible) == len(model.classifier.classes_)
