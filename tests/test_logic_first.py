import pytest
from codeccat.logic_first import evaluate, _validate, SUPPORTED

@pytest.mark.parametrize("codec,text",[
    ("utf-8","日本語の文章"),
    ("shift-jis","日本語の文章"),
    ("euc-jp","日本語の文章"),
    ("big5","繁體中文測試"),
    ("gb18030","中文编码测试"),
])
def test_known_encoded_text_is_not_rejected(codec,text):
    # Valid examples should never be marked invalid by structural parser.
    raw=text.encode({"shift-jis":"shift_jis","euc-jp":"euc_jp"}.get(codec,codec))
    assert evaluate(raw)[codec].status=="valid"

def test_trailing_partial_sequences_report_incomplete():
    assert _validate(bytes.fromhex("82"),"shift-jis").status=="incomplete"
    assert _validate(bytes.fromhex("8fa1"),"euc-jp").status=="incomplete"
    assert _validate(bytes.fromhex("813081"),"gb18030").status=="incomplete"
    assert _validate(bytes.fromhex("e381"),"utf-8").status=="incomplete"

def test_bad_sequences_are_invalid():
    assert _validate(bytes.fromhex("80"),"shift-jis").status=="invalid"
    assert _validate(bytes.fromhex("8f40"),"euc-jp").status=="invalid"
    assert _validate(bytes.fromhex("817f"),"big5").status=="invalid"
    assert _validate(bytes.fromhex("81308140"),"gb18030").status=="invalid"
    assert _validate(bytes.fromhex("c0af"),"utf-8").status=="invalid"

def test_ascii_is_ambiguous_not_unique():
    out=evaluate(b"plain ascii")
    assert all(x.status=="valid" for x in out.values())
    assert set(out)==set(SUPPORTED)
