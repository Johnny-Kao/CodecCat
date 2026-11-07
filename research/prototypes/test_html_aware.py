"""Research-only opt-in HTML decoding prototype contract tests.

No mutation of CodecCat. Test proposed behavior, explicit failure cases, and
the known CPython-vs-WHATWG text-mapping caveat.
"""
from __future__ import annotations
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path("research/prototypes").resolve()))
from html_aware import decode_html_prototype,HtmlDecodeError

class HtmlPrototypeTests(unittest.TestCase):
    def test_real_gbk_meta(self):
        raw=b'<meta charset="gbk"><p>\xd6\xd0\xce\xc4</p>'
        v=decode_html_prototype(raw)
        self.assertEqual((v.encoding,v.source),("gbk","meta"))
        self.assertIn("中文",v.text)
        self.assertFalse(v.used_replacement)

    def test_http_equiv(self):
        raw=b'<meta http-equiv="Content-Type" content="text/html; charset=gbk"><p>\xd6\xd0\xce\xc4</p>'
        self.assertEqual(decode_html_prototype(raw).encoding,"gbk")

    def test_comments_scripts_attributes_do_not_set_encoding(self):
        cases=(
            b'<!-- <meta charset=gbk> --><p>Hello</p>',
            b'<script>var a="<meta charset=gbk>";</script><p>Hello</p>',
            b'<div title="<meta charset=gbk>">Hello</div>',
        )
        for raw in cases:
            with self.subTest(raw=raw):
                v=decode_html_prototype(raw)
                self.assertEqual((v.encoding,v.source),("windows-1252","fallback"))

    def test_late_genuine_meta(self):
        raw=b"X"*1024+b"<meta charset=gbk><p>\xd6\xd0</p>"
        self.assertEqual(decode_html_prototype(raw).encoding,"gbk")

    def test_precedence_bom_transport_override(self):
        raw=b'<meta charset="gbk"><p>Hello</p>'
        self.assertEqual(decode_html_prototype(raw,content_type="text/html; charset=utf-8").source,"transport")
        self.assertEqual(decode_html_prototype(raw,override_encoding="windows-1251",
                         content_type="text/html; charset=utf-8").source,"override")
        v=decode_html_prototype(b'\xef\xbb\xbf'+raw,override_encoding="windows-1251",
                                content_type="text/html; charset=gbk")
        self.assertEqual((v.source,v.encoding),("bom","utf-8"))
        self.assertFalse(v.text.startswith("\ufeff"))

    def test_web_aliases(self):
        self.assertEqual(decode_html_prototype(b'<meta charset=gb2312>\xd6\xd0').encoding,"gbk")
        self.assertEqual(decode_html_prototype(b'<meta charset=windows-31j>').encoding,"shift_jis")
        self.assertEqual(decode_html_prototype(b'<meta charset=iso-8859-1>').encoding,"windows-1252")
        self.assertEqual(decode_html_prototype(b'<meta charset=utf-16>').encoding,"utf-8")

    def test_strict_vs_replacement(self):
        raw=b"<meta charset=utf-8>\xff"
        v=decode_html_prototype(raw)
        self.assertTrue(v.used_replacement)
        self.assertIn("\ufffd",v.text)
        with self.assertRaises(HtmlDecodeError):
            decode_html_prototype(raw,errors="strict")

    def test_caller_controlled_limits_and_types(self):
        with self.assertRaises(TypeError):decode_html_prototype("not bytes")
        with self.assertRaises(HtmlDecodeError):decode_html_prototype(b"x"*12,max_bytes=10)
        with self.assertRaises(HtmlDecodeError):decode_html_prototype(b"foo",override_encoding="unknown-123")
        with self.assertRaises(HtmlDecodeError):decode_html_prototype(b"foo",content_type="text/html\ncharset=gbk")
        with self.assertRaises(ValueError):decode_html_prototype(b"foo",max_scan_bytes=0)
        with self.assertRaises(ValueError):decode_html_prototype(b"foo",errors="ignore")

    def test_duplicate_meta_first_valid_wins(self):
        raw=b'<meta charset=not-real><meta charset=gbk><meta charset=utf-8>'
        self.assertEqual(decode_html_prototype(raw).encoding,"gbk")

    def test_runtime_is_not_required_for_html_prototype(self):
        import sys
        self.assertNotIn("numpy",sys.modules)
        self.assertNotIn("sklearn",sys.modules)


if __name__=="__main__":unittest.main()
