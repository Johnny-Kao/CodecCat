import pytest

from codeccat.vnext_l3_language import LanguageCoherence, SCHEMA
from codeccat.vnext_strong_l2 import Ranked


def samples():
    return [
        (("The café has a résumé and pâtisserie. " * 30).encode("utf-8"),
         "utf-8", "french-a"),
        (("A naïve façade and jalapeño in the café. " * 30).encode("utf-8"),
         "utf-8", "french-b"),
        (("Ordinary English language sample. " * 30).encode("latin1"),
         "latin1", "english-a"),
    ]


def test_language_model_can_correct_mojibake_only_when_l2_is_ambiguous():
    model = LanguageCoherence.fit(samples())
    raw = ("café résumé pâtisserie " * 15).encode("utf-8")
    ranked = Ranked("latin1", ("latin1", "utf-8"), (0.50, 0.45), None, False)
    output = model.rerank(ranked, raw, min_language_advantage=0.0)
    assert output.called
    assert output.changed
    assert output.encoding == "utf-8"
    assert output.language_advantage > 0
    far = Ranked("latin1", ("latin1", "utf-8"), (1.0, 0.0), None, False)
    skipped = model.rerank(far, raw)
    assert not skipped.called
    assert skipped.encoding == "latin1"


def test_same_text_is_underdetermined_and_cannot_be_reranked():
    model = LanguageCoherence.fit(samples())
    ranked = Ranked("latin1", ("latin1", "utf-8"), (0.5, 0.4), None, False)
    result = model.rerank(ranked, b"Plain ASCII text repeated many times.")
    assert result.encoding == "latin1"
    assert result.reason == "identical-decoded-text"
    assert not result.called


def test_bom_fastpath_and_invalid_candidate_not_overridden():
    model = LanguageCoherence.fit(samples())
    assert model.rerank(Ranked("utf-8", ("utf-8",), (1.0,), "utf-8-sig", False), b"\xef\xbb\xbfSome text").called is False
    raw = b"long stream with invalid utf8: \xff" * 5
    ranked = Ranked("latin1", ("latin1", "utf-8"), (0.8, 0.7), None, False)
    result = model.rerank(ranked, raw)
    assert result.encoding == "latin1"
    assert not result.changed


def test_save_load_and_schema_validation(tmp_path):
    model = LanguageCoherence.fit(samples())
    path = tmp_path / "l3.json"
    model.save(path)
    loaded = LanguageCoherence.load(path)
    assert loaded.schema == SCHEMA
    assert loaded == model
    assert loaded.score("café résumé pâtisserie " * 20) == model.score("café résumé pâtisserie " * 20)
    path.write_text(path.read_text().replace(SCHEMA, "future-schema"))
    with pytest.raises(ValueError, match="schema"):
        LanguageCoherence.load(path)


def test_no_training_and_inadequate_grams():
    with pytest.raises(ValueError, match="no decodable"):
        LanguageCoherence.fit([])
    model = LanguageCoherence.fit(samples())
    assert model.score("hi") is None
