"""BOM semantics and end-to-end L1/L2 output contract."""
import pytest
from codeccat.vnext_decoding import decode
from codeccat.vnext_strong_l2 import TrainableDetector

@pytest.fixture(scope="module")
def detector():
    samples=[
        (("Japanese words and explanation 日本語 " * 24).encode("utf-8"),"utf-8","jp"),
        (("Another Unicode text with café " * 24).encode("utf-8"),"utf-8","latin"),
        (("Ordinary English sentence. "*24).encode("utf-16-le"),"utf-16-le","le"),
        (("Second English statement. "*24).encode("utf-16-be"),"utf-16-be","be"),
        (("UTF-32 representation. "*24).encode("utf-32-le"),"utf-32-le","u32le"),
        (("Different UTF-32 text. "*24).encode("utf-32-be"),"utf-32-be","u32be"),
        (("A UTF-8 signature header. "*24).encode("utf-8-sig"),"utf-8-sig","sig"),
        (("UTF-16 generic text. "*24).encode("utf-16"),"utf-16","generic16"),
        (("UTF-32 generic text. "*24).encode("utf-32"),"utf-32","generic32"),
    ]
    return TrainableDetector.fit(samples,feature_profile="experimental13")

def test_utf8_explicit_bom_contract(detector):
    raw=b"\xef\xbb\xbfHello world"
    keep=decode(detector,raw,bom_policy="preserve")
    strip=decode(detector,raw,bom_policy="strip")
    assert keep.bom_encoding=="utf-8-sig"
    assert keep.text=="\ufeffHello world"
    assert strip.text=="Hello world"
    assert keep.encoding==strip.encoding

@pytest.mark.parametrize("codec,signature",[
    ("utf-16-le",b"\xff\xfe"),("utf-16-be",b"\xfe\xff"),
    ("utf-32-le",b"\xff\xfe\x00\x00"),("utf-32-be",b"\x00\x00\xfe\xff")
])
def test_unicode_endian_bom_output_separated_from_codec_label(detector,codec,signature):
    raw=signature+"Text written here".encode(codec)
    a=decode(detector,raw,bom_policy="preserve")
    b=decode(detector,raw,bom_policy="strip")
    assert a.bom_present and b.bom_present
    assert a.text=="\ufeffText written here"
    assert b.text=="Text written here"

def test_no_bom_does_not_strip_interior_codepoints(detector):
    raw=b"A regular English text."
    a=decode(detector,raw,bom_policy="preserve")
    b=decode(detector,raw,bom_policy="strip")
    assert a.text==b.text
    raw_utf16=("A\ufeffB").encode("utf-16-le")
    c=decode(detector,raw_utf16,bom_policy="strip")
    if c.accepted:
        assert "\ufeff" in c.text

def test_invalid_policy_and_type(detector):
    with pytest.raises(TypeError):
        decode(detector,"not bytes")
    with pytest.raises(ValueError,match="bom_policy"):
        decode(detector,b"Hello",bom_policy="guess")

def test_integrated_L1_selection_and_lazy_L2_have_same_output(detector):
    for data in [
        b"\xef\xbb\xbfUTF-8 BOM", "Testing unicode".encode("utf-16"),
        "Testing UTF32".encode("utf-32"), b"some undecidable ASCII",
        ("Rocket " * 100).encode("utf-8"),
    ]:
        for policy in ("preserve","strip"):
            assert decode(detector,data,bom_policy=policy)==decode(
                detector,data,bom_policy=policy,structural_first=True)


def test_prefix_cannot_hide_invalid_trailing_bytes(detector):
    # L2 may rank from the first 4096 bytes, but text decoding must validate
    # the full input instead of claiming success for an invalid late suffix.
    raw=b"\xef\xbb\xbf"+b"ASCII text "*500+b"\xff"
    assert len(raw)>4096
    result=decode(detector,raw,bom_policy="strip")
    assert result.encoding is None
    assert result.text is None
    assert result.ambiguous
    assert result.reason=="invalid_selected_codec"

def test_unknown_bom_family_abstains_without_substituting_encoding():
    m=TrainableDetector.fit([
        (b"UTF-8 paragraph"*30,"utf-8","a"),
        (("日本語とテキスト"*30).encode("shift_jis"),"shift_jis","b"),
    ])
    result=decode(m,"Not supported UTF-32".encode("utf-32"))
    assert not result.accepted
    assert result.encoding is None
    assert result.text is None
