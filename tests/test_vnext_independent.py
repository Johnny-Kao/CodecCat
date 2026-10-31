from codeccat.vnext_independent import fit,vector,SCHEMA
import numpy as np
import pytest

def test_order_matters():
    assert not np.array_equal(vector(bytes.fromhex("81408142")),vector(bytes.fromhex("40814281")))

def test_user_trainable_without_r21():
    texts=[("日本語ひらがなカタカナ"*12).encode("shift_jis"),
           ("日本語漢字の文章"*12).encode("shift_jis")]
    utf=[("Hello world français "*20).encode(),("Other latin café "*20).encode()]
    model=fit([(x,"shift-jis") for x in texts]+[(x,"utf-8") for x in utf])
    assert model.schema==SCHEMA
    assert model.rank(texts[0])
    assert set(model.rank(texts[0]))=={"utf-8","shift-jis"}

def test_requires_two_encoding_classes():
    with pytest.raises(ValueError):
        fit([(b"abc","utf-8")])
