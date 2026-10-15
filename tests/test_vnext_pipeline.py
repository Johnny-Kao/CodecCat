import tempfile
from pathlib import Path
from codeccat.vnext_pipeline import train,Model,SCHEMA

def test_user_data_roundtrip():
    sample=[
        (("日本語の文章"*20).encode("shift_jis"),"shift_jis","ja-a"),
        (("別の日本語文章"*20).encode("shift_jis"),"shift_jis","ja-b"),
        (("English text café "*20).encode("utf-8"),"utf-8","en-a"),
        (("Unicode example text "*20).encode("utf-8"),"utf-8","en-b"),
    ]
    m=train(sample)
    assert m.schema==SCHEMA
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/"model.json";m.save(path)
        same=Model.load(path)
        for data,_,_ in sample:
            assert same.detect(data)==m.detect(data)

def test_ignores_invalid_candidate_without_legacy_model():
    m=train([
      (b"\xff\xfe\x00\x00H\x00\x00\x00","utf-32","source1"),
      (("Hello café "*20).encode(),"utf-8","source2"),
    ])
    r=m.detect(b"\xff\xfe\x00\x00H\x00\x00\x00")
    assert "utf-8" not in r.candidates

def test_no_training_set():
    import pytest
    with pytest.raises(ValueError):train([])

def test_sampling_cut_does_not_exclude_utf8_with_partial_terminal_sequence():
    # UTF-8 starts a multi-byte sequence at byte 4095; full input is valid.
    data = b"A" * 4095 + "日".encode("utf-8") + b"tail"
    assert len(data) > 4096
    m = train([
        (("日本語"*20).encode("utf-8"),"utf-8","ja"),
        (("Cyrillic example "*20).encode("cp1251"),"cp1251","ru"),
    ])
    result = m.detect(data)
    assert "utf-8" in result.candidates
