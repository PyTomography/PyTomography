# Security policy

## Reporting a vulnerability

Please don't report security problems in public issues, pull requests or Discussions.

Report them privately through GitHub's private vulnerability reporting: open the repository's
[Security tab](https://github.com/PyTomography/PyTomography/security) and choose **Report a vulnerability**
([direct link](https://github.com/PyTomography/PyTomography/security/advisories/new)). Only the maintainers can see
the report.

Please include what is affected, how to reproduce it, and what an attacker could do with it. We will reply within five
working days, keep you updated while we fix it, and credit you in the advisory unless you would rather we didn't.

## Supported versions

Security fixes go into the latest release, as a patch release (for example 4.0.1). Older versions are not updated.

## Scope

PyTomography reads files such as DICOM, Interfile and GATE/ROOT output, and downloads tutorial data. A file that makes
it run code, read or write outside the paths it was given, or use unbounded memory is in scope. Wrong reconstructed
values are bugs, not vulnerabilities: please report them as issues.
