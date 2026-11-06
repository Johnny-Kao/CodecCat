# Security policy

## Supported versions

| Version | Security support |
| --- | --- |
| `0.1.x` (current Stable line) | Best-effort fixes and coordinated disclosure |
| Earlier `0.1.0a*` prereleases | Not actively supported; upgrade to Stable |

No response-time SLA or automatic backport to prerelease versions is promised.

## Reporting privately

**Do not open a public issue or PR with exploit details, proof-of-concept payloads, private documents, secrets or security-sensitive test data.**

1. Open [CodecCat Security Advisories](https://github.com/Johnny-Kao/CodecCat/security/advisories) and select **Report a vulnerability** if private reporting is enabled.
2. If that option is unavailable, [open a non-sensitive Issue](https://github.com/Johnny-Kao/CodecCat/issues/new/choose) asking the maintainer for a **private reporting channel**. Do not include vulnerability details in that Issue.
3. Include the affected version, impact, reproduction steps, affected entry points, and any safe evidence through the private channel.

The maintainer will review reports, coordinate fixes and attribution where appropriate, and request reasonable time for disclosure. This policy does not promise a deadline or compensation.

## Security scope

Relevant issues include model parser memory-safety/resource exhaustion, malicious Wheel/model supply-chain risks, untrusted byte processing, incorrect security-sensitive defaults, and secret exposure. Encoding inference itself cannot prove the source encoding; applications must not treat a guessed codec as an authentication or trust boundary.

For third-party training-data rights, use [RIGHTS_AND_PROVENANCE.md](RIGHTS_AND_PROVENANCE.md), not this vulnerability reporting route.
