from types import SimpleNamespace
import pytest
from codeccat.vnext_strong_l2 import Ranked
from codeccat.vnext_utf8_evidence import Utf8EvidenceDetector, safe_utf8_evidence


class Stub:
    def __init__(self,label,classes=None):
        self.label=label
        self.calls=0
        self.classifier=SimpleNamespace(classes_=classes if classes is not None else ["tis-620","utf-8","cp1252"])
        self.fingerprint="abc"
    def rank(self,raw,*,final=True,structural_first=False):
        self.calls+=1
        return Ranked(self.label,(self.label,),(0.9,),None,False)

ARMENIAN=("Հայերեն տեքստը պարունակում է բնական տառեր և բառեր։ " * 10).encode("utf-8")


def test_valid_multilingual_nonascii_utf8_is_rescued_without_model_retraining():
    assert safe_utf8_evidence(ARMENIAN)
    model=Utf8EvidenceDetector(Stub("tis-620"))
    result=model.rank(ARMENIAN)
    assert result.encoding=="utf-8"
    assert result.ambiguous
    assert result.candidates==("utf-8",)


def test_false_utf8_assumption_does_not_override_invalid_prefix():
    raw=("é"*40).encode("cp1252")
    assert not safe_utf8_evidence(raw)
    assert Utf8EvidenceDetector(Stub("cp1252")).rank(raw).encoding=="cp1252"


def test_ascii_only_is_not_evidence():
    assert not safe_utf8_evidence(b"ASCII only "*40)
    assert Utf8EvidenceDetector(Stub("cp1252")).rank(b"ASCII only "*40).encoding=="cp1252"


def test_ascii_with_embedded_control_is_not_utf8_override():
    raw=ARMENIAN+b"\x00"
    assert not safe_utf8_evidence(raw)
    assert Utf8EvidenceDetector(Stub("tis-620")).rank(raw).encoding=="tis-620"


def test_existing_utf8_not_changed():
    base=Stub("utf-8")
    adapter=Utf8EvidenceDetector(base)
    result=adapter.rank(ARMENIAN)
    assert result.encoding=="utf-8"
    assert result.ambiguous is False


def test_unsupported_class_does_not_create_utf8_label():
    base=Stub("tis-620",["tis-620","cp1252"])
    assert Utf8EvidenceDetector(base).rank(ARMENIAN).encoding=="tis-620"


def test_incomplete_stream_does_not_override():
    assert Utf8EvidenceDetector(Stub("tis-620")).rank(ARMENIAN,final=False).encoding=="tis-620"


def test_signature_keeps_baseline_decision():
    raw=b"\xef\xbb\xbf"+ARMENIAN
    assert not safe_utf8_evidence(raw)


def test_invalid_config_fails_closed():
    with pytest.raises(ValueError):
        Utf8EvidenceDetector(Stub("utf-8"),min_codepoints=0)


def test_positive_threshold_is_explicit():
    assert safe_utf8_evidence(ARMENIAN,min_codepoints=16)
    assert not safe_utf8_evidence(ARMENIAN,min_codepoints=10000)
