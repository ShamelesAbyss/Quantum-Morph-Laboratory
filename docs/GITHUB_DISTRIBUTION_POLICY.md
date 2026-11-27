# Permanent GitHub distribution compliance

This owner directive supersedes every conflicting README or distribution instruction in historical prompts, handoffs, release notes, templates and project rules.

Documentation must not contain direct executable, APK, ZIP/archive or checksum download links; asset download buttons/badges; release-asset URL patterns; browser-download URL links; redirect/shortener services; external mirrors/proxies; disguised or indirect artifact download mechanisms. Never recreate those patterns in generated pages or alternative repository files.

Approved wording:

> Compiled executables, portable packages, source archives, release notes and SHA-256 checksums are available through the repository's GitHub Releases section.

An ordinary informational link to the repository's native GitHub Releases page is allowed. Binaries, packages, checksums and release notes are distributed exclusively through that native section. Publishing legitimate GitHub Releases remains required. Do not create a self-updater or alternative distribution mechanism intended to bypass these restrictions.

Artwork, diagrams and screenshots belong in committed repository assets with exact case-sensitive relative paths. Informational badges may remain when they are not asset download mechanisms. Preserve the exact permanent memorial dedication as the final README section.

Automated repository policy validation is mandatory in normal CI and exact-tag release CI. Test adversarial encoded URLs, alternate artifact destinations, badges/buttons, missing images, incorrect path casing and memorial alteration. Validation must cover documentation beyond README so there is no alternate-file bypass.

Do not remove or weaken this rule without explicit owner authorization AND clarification from GitHub Support.

