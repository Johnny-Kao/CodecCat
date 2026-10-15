from codeccat.vnext_bom import inspect_bom

def test_longest_bom_precedence():
    cases=[
        (b"\xff\xfe\x00\x00A\x00\x00\x00","utf-32-le"),
        (b"\x00\x00\xfe\xff\x00\x00\x00A","utf-32-be"),
        (b"\xff\xfeA\x00","utf-16-le"),
        (b"\xfe\xff\x00A","utf-16-be"),
        (b"\xef\xbb\xbfHello","utf-8-sig"),
    ]
    for data,expected in cases:
        assert inspect_bom(data).encoding==expected

def test_streaming_prefix_does_not_force_utf16():
    assert inspect_bom(b"\xff\xfe",final=False).incomplete
    assert inspect_bom(b"\xff\xfe",final=True).encoding=="utf-16-le"
    assert inspect_bom(b"\xff\xfe\x00",final=False).incomplete
    assert inspect_bom(b"\x00\x00",final=False).incomplete

def test_bom_is_only_evidence():
    assert inspect_bom(b"123").encoding is None
    assert inspect_bom(b"").encoding is None
