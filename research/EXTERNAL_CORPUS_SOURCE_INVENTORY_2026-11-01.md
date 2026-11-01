# External encoding corpus candidates — 2026-11-01

Status: **discovery and source review**, not dataset acceptance and not training.

## Verified repository paths

| Candidate | Path | Kind | Immediate verdict |
|---|---|---|---|
| BYVoid/uchardet | `test/hu/iso-8859-2.txt` | Hungarian legacy-encoded test fixture | Fetch original blob and validate actual encoding, source and rights |
| BYVoid/uchardet | `test/hu/windows-1250.txt` | Hungarian comparison fixture | Same conditions; do not assume independent source from separate codec filename |
| BYVoid/uchardet | `test/{ar,bg,da,de,el,en,eo,es,fr,he,hu,ja,ko,ru,th,tr,vi,zh}/` | Language-grouped legacy fixtures | Inventory original byte data and file lineage first |
| jawah/charset_normalizer | `data/`, `data/NOTICE.md` | 17 multilingual regression samples | NOTICE says Wikipedia CC-BY-SA or public-domain archive; file-level provenance/attribution needs review |
| unicode-org/icu | `icu4c/source/test/testdata/csdetest.xml` | Charset detection conformance data | Use as engineering conformance reference, not blind natural-source confirmation |
| web-platform-tests/wpt | `encoding/` | Encoding standard conformance tests | Correctness oracle, not natural document domain validation |
| unicode-org/icu-data | `charset/` | Conversion mapping tables | Codec mapping oracle, not natural document corpus |
| Zaneham/cobol-ebcdic | `test/TESTEBCD.cbl` | CP037/CP500 conversion tests, project Apache-2.0 | Useful tests, not source-diverse EBCDIC natural documents |

Sources:
- https://github.com/BYVoid/uchardet/tree/master/test
- https://github.com/jawah/charset_normalizer/tree/master/data
- https://github.com/jawah/charset_normalizer/blob/master/data/NOTICE.md
- https://github.com/unicode-org/icu/tree/main/icu4c/source/test/testdata
- https://github.com/web-platform-tests/wpt/tree/master/encoding
- https://github.com/unicode-org/icu-data/tree/main/charset
- https://github.com/Zaneham/cobol-ebcdic

## Critical acceptance caveats

1. Test fixtures from third-party detectors may derive from common training sources, may be synthetic or transcodes, and may duplicate prior benchmark material. Repo diversity is not document-source independence.
2. License on source code is not automatically a license on the embedded text. The charset-normalizer NOTICE explicitly distinguishes Wikipedia CC-BY-SA and public-domain materials.
3. GitHub's text rendering is **not** a reliable way to validate the original legacy encoding. Download pinned blob bytes in GitHub Actions, then verify file encoding, checksum and exact-byte/document lineage before any model training.
4. A legacy encoding specified in a filename is not verified ground truth unless compared against author metadata or an authentic reference.
5. In particular, do not count parallel ISO-8859-2 / CP1250 encoded versions of the same Hungarian passage as independent source documents.
6. No authentic naturally occurring independent CP037/CP500 documents with trustworthy original-encoding provenance have yet been identified here; conversion code and synthesized fixtures are not substitutes.
7. Do not upload copyrighted full text as public GitHub Actions artifacts. Retain only permitted metadata, hashes, counts and aggregate validation outputs.

## Execution decision

First inventory **file-level byte hashes, label metadata, source lineage and license evidence** using pinned revisions. Quarantine unverifiable sources. Candidate fixtures may serve a development diagnostic, but only properly evidenced real original documents can enter the untouched confirmation cohort. Keep all model training, production and alpha tags unchanged.
