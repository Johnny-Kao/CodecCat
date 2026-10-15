"""Deterministic gate contract; these do not count as a research batch."""
import codecs
from codeccat.vnext_gate import evaluate

def test_bom_utf32_is_never_forced_into_utf16():
    value = "Hello!".encode("utf-32")
    r = evaluate(value, ("utf-16", "utf-32", "utf-8"))
    assert r.bom.encoding == "utf-32-le" or r.bom.encoding == "utf-32-be"
    assert "utf-32" in r.possible

def test_invalid_utf8_excluded_only_with_proof():
    x = evaluate(b"\xff\xfe", ("utf-8", "latin1"))
    assert "utf-8" in x.impossible
    assert "latin1" in x.possible

def test_partial_utf8_kept_until_final():
    x = evaluate(b"\xe3\x81", ("utf-8",), final=False)
    assert "utf-8" in x.possible
    y = evaluate(b"\xe3\x81", ("utf-8",), final=True)
    assert "utf-8" in y.impossible

def test_unknown_codec_not_hard_eliminated():
    x = evaluate(b"abc", ("future-standard-xyz",))
    assert x.possible == ("future-standard-xyz",)
    assert x.incomplete == ("future-standard-xyz",)

def test_alias_variants():
    for e in ("utf-8", "shift_jis", "euc_jp", "gb18030", "big5", "utf-16"):
        raw = "abc".encode(e)
        assert e in evaluate(raw, (e,)).possible
