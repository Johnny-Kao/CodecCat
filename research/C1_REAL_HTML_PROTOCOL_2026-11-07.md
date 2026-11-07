# Preregistered historical natural HTML evaluation — C1 warning-only diagnostic

Registered before evaluating any detector predictions on this HTML corpus. No default Runtime change or PyPI release is authorized.

Source: independent IUST-HTMLCharDet crawler archive at commit 0a7f382e2d488d46f1e12bad647f0384c4a2288f; eval/test-data/encoding-wise/corpus.zip, Git blob 9d6ff0ff9c65c5afc96e1a10b996a51e7c4d3ba5 (37,123,043 bytes). Inventory-only CI #42343274424 counted 2,675 real web-crawled HTML files: GBK 419, Shift_JIS 640, Windows-1251 314, Windows-1256 645, UTF-8 657. Raw source stays in ephemeral Actions checkout.

Frozen Staging model SHA256: 16073306b5027bf93acc2883f57fbda0cb0cf2d77b9a7bb62cf1b07fcbb9748b. Historical training source: chardet/test-data commit b0c0d206913ba118ffea2f7d306883e7a4a76db7, verified manifest SHA256 8ec3007dbd59336cee6018cfa44aeded39cca205e93f5fc9f24ee5ecf87f3ca9, 3,086 input files.

Selection fixed before scoring: iterate all ZIP members sorted by path; reject unsafe paths, unknown codec labels, directories, oversized (>6MB) or empty files, duplicate full-input SHA256, and sources failing strict Unicode decoding with their directory label. ZIP maximum 500MB uncompressed and 10,000 entries. Count exclusions by label/reason, never select on detector outcomes. No content normalization or truncation.

Overlap gate BEFORE predictions: compare whole-file SHA256, first-4096-byte SHA256, and sampled 128-byte block overlap (>=3 matching blocks) against every pinned historical training file. Exclude any matched/candidate sample and report counts. Negative byte screen is not proof of semantic independence.

Metrics, separately per encoding family: exact Unicode agreement with a strict decode under directory-provided codec label (PROXY, not verified human-labeled true text); baseline incorrect; warning counts split into baseline label-agreeing and label-disagreeing; samples not flagged. Confirm diagnostic never mutates returned Detection fields; measure extra diagnostic-only p50/p95 time in microseconds and do not equate to end-to-end latency. Record shared CI noise and correlated crawling bias.

False-positive rates against directory labels are not reliable population false-positive rates: HTML labels may be inaccurate, web crawler projects are correlated, and Windows/ISO browser label semantics differ from strict Python decoding. Store only aggregate counts, hashes, file paths and metadata; never raw bytes, text, model artifact or extracted HTML in uploaded artifacts.

No Runtime/default behavior/model/API/PyPI changes. Diagnostic remains opt-in in Draft staging PR #47; human review and license/provenance check precede release.