from codeccat.vnext_strong_l2 import Ranked
from codeccat.vnext_selective import SelectiveLatinDetector, _family_eligible


class FakeModel:
    def __init__(self, encoding, *, bom=None, candidates=None):
        self.value = encoding
        self.bom = bom
        self.calls = 0
        self.candidates = tuple(candidates or [encoding])
        self.fingerprint = "test"
        self.classifier = object()
    def rank(self, data, *, final=True, structural_first=False):
        self.calls += 1
        return Ranked(self.value, self.candidates, tuple(1.0 for _ in self.candidates), self.bom, False)


def test_latin_family_is_canonical_and_generic():
    assert _family_eligible("cp1252")
    assert _family_eligible("iso-8859-2")
    assert not _family_eligible("cp037")
    assert not _family_eligible("utf-8")
    assert not _family_eligible(None)


def test_unicode_routing_is_never_changed():
    base = FakeModel("utf-8")
    alt = FakeModel("cp1252")
    assert SelectiveLatinDetector(base, alt).rank(b"hello").encoding == "utf-8"
    assert alt.calls == 0


def test_ebcdic_routing_is_never_changed():
    base = FakeModel("cp037")
    alt = FakeModel("cp500")
    assert SelectiveLatinDetector(base, alt).rank(b"hello").encoding == "cp037"
    assert alt.calls == 0


def test_latin_correction_requires_different_unicode_text():
    raw = b"Here is \x80 to distinguish the byte mappings"
    base = FakeModel("iso8859-1", candidates=["iso8859-1","cp1252"])
    alt = FakeModel("cp1252", candidates=["cp1252","iso8859-1"])
    result = SelectiveLatinDetector(base, alt).rank(raw)
    assert result.encoding == "cp1252"
    assert result.candidates[0] == "cp1252"


def test_equivalent_decoded_text_keeps_baseline_label():
    raw = b"ASCII only phrase"
    base = FakeModel("iso8859-1")
    alt = FakeModel("cp1252")
    assert SelectiveLatinDetector(base, alt).rank(raw).encoding == "iso8859-1"


def test_bom_does_not_allow_specialist_override():
    raw = b"\xef\xbb\xbf" + b"Nonunicode \x80 with byte signature"
    base = FakeModel("iso8859-1")
    alt = FakeModel("cp1252")
    assert SelectiveLatinDetector(base, alt).rank(raw).encoding == "iso8859-1"
    assert alt.calls == 0


def test_invalid_specialist_codec_cannot_override():
    raw = b"Short \x81 non-ASCII input"
    base = FakeModel("iso8859-1")
    alt = FakeModel("cp1252")
    assert SelectiveLatinDetector(base, alt).rank(raw).encoding == "iso8859-1"


def test_missing_alt_label_cannot_override():
    raw = b"Data \x80 raw"
    base = FakeModel("iso8859-1")
    alt = FakeModel(None, candidates=[])
    assert SelectiveLatinDetector(base, alt).rank(raw).encoding == "iso8859-1"
