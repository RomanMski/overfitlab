# Security policy

## Supported versions

StressFold is pre-1.0 software. Security fixes are applied to the current `0.3.x` line and the default branch. Older snapshots are not maintained.

## Reporting a vulnerability

Use GitHub’s private vulnerability-reporting form from the repository’s **Security** tab. Please include the affected version or revision, impact, a minimal reproduction, and any suggested mitigation. Remove credentials, personal data, and proprietary datasets from the report.

If private reporting is not enabled, open a public issue that asks for a confidential maintainer contact. Do not include exploit details in that issue.

Maintainers will acknowledge the report when it is seen, assess scope, and coordinate a fix and disclosure where appropriate. No fixed response-time guarantee is offered while the project is maintained by volunteers.

## Data handling

The Python package reads data from the calling process and writes only to paths requested by the caller. The browser lab parses the selected CSV in browser memory, and the current implementation has no dataset-upload endpoint. Self-contained reports and exported variants can still contain sensitive metadata or values, so treat them under the same access controls as the source data.

Incorrect statistical conclusions, undocumented assumptions, or reproducibility failures are important bugs but are normally not security vulnerabilities. Report those through the regular issue tracker unless confidential data or an exploitable condition is involved.
