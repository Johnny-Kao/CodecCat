"""Regression tests for opt-in layered runtime BOM policy.
Run only via research CI; no production runtime changes.
"""
from types import SimpleNamespace
from codeccat.vnext_layered_pipeline import detect

class FixedModel:
    def __init__(self, encoding):
        self.encoding = encoding
    def rank(self, raw, *, final):
        assert final is True
        return SimpleNamespace(encoding=self.encoding, candidates=(self.encoding,),
                               scores=(1.0,), bom_hint=None)

def check():
    utf7=FixedModel("utf-7")
    for text in ("\ufeffPlain ASCII", "\ufeffRésumé français",
                 "\ufeff中文テスト"):
        raw=text.encode("utf-7")
        assert raw.startswith((b"+/v8-", b"+/v9-", b"+/v+-", b"+/v/-")), raw[:10]
        stripped=detect(utf7,raw,reject_binary=False,bom_policy="strip")
        preserved=detect(utf7,raw,reject_binary=False,bom_policy="preserve")
        assert stripped.accepted and stripped.text==text[1:], (stripped,text)
        assert preserved.accepted and preserved.text==text, (preserved,text)
    # Ordinary text containing the UTF-7 shifted BOM later in the stream is
    # content rather than an initial signature and must not be dropped.
    raw="hello\ufeff world".encode("utf-7")
    assert detect(utf7,raw,reject_binary=False,bom_policy="strip").text=="hello\ufeff world"
    # UTF-7-looking ASCII bytes are not a BOM under another chosen codec.
    ascii_sample=b"+/v8-Plain ASCII"
    assert detect(FixedModel("utf-8"),ascii_sample,reject_binary=False,bom_policy="strip").text==ascii_sample.decode()
    # Preserve prior UTF-8 BOM handling.
    raw=b"\xef\xbb\xbfHello"
    assert detect(FixedModel("utf-8"),raw,reject_binary=False,bom_policy="strip").text=="Hello"
    assert detect(FixedModel("utf-8"),raw,reject_binary=False,bom_policy="preserve").text=="\ufeffHello"
    print("UTF7_BOM_POLICY_REGRESSION_PASS")

if __name__=="__main__":
    check()
