# Copyright, training data and rights inquiries

CodecCat is a character-encoding detector. Its published source code is
licensed under the repository's MIT LICENSE. This license applies to
CodecCat's original code, **not** to third-party training texts.

## Trained-model provenance

The current experimental default model was fitted using a pinned revision of
[chardet/test-data](https://github.com/chardet/test-data), commit
`b0c0d206913ba118ffea2f7d306883e7a4a76db7`.
The upstream test-data catalog identifies mixed origins, including CulturaX,
mC4/OSCAR/Common Crawl-derived text, test suites from other detector
projects, web captures and community-contributed samples.
Upstream states that the individual test files remain copyrighted by their
respective publishers. No universal license is asserted for that collection.

The default CCM2 file is a numeric inference model containing encoding
class labels, vocabulary n-grams of up to three Unicode characters,
IDF values, classifier intercepts and weights, plus an integrity digest.
The optional L3 JSON contains trigram counts, numeric statistics and
provenance fingerprint. These are derived statistical artifacts, not a
distribution of the original training files. This description is technical,
**not** a representation that all legal questions regarding model training
or distribution have been settled.

The model fingerprint and checksum help verify integrity and provenance.
They do not establish a license or authenticate a third-party copyright claim.

## Rights inquiries and removal requests

If you believe a CodecCat distribution contains or reproduces copyrighted
material you own, or that a training-data source was used in a way that
infringes your rights, please contact the maintainer:

**[Open a rights inquiry in CodecCat GitHub Issues](https://github.com/Johnny-Kao/CodecCat/issues/new)**

Please identify the material or source, explain your rights and specific
concern, and provide a way to follow up. Do **not** post confidential
documents, private identity records or personal information in a public issue.
For sensitive claims, use an issue containing only a request for a private
contact channel; do not disclose sensitive evidence in the public thread.

The maintainer will review substantiated requests and, where warranted,
correct attribution, remove affected material, retrain a model using
alternative data, or suspend distribution of an affected artifact. This is
a contact-and-review procedure, not a statutory safe harbor or waiver of
anyone's rights. No fixed response deadline is promised.

## Release audit scope

The release packaging audit checks the distributions for unintended raw
training samples and verifies the expected short-gram model structure.
It is not a general-purpose copyright clearance, proof of non-memorization
or independent legal opinion.
