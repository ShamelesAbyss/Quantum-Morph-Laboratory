# Security and data handling

Do not submit credentials, authentication tokens, private model weights, personal information, or private experiment data to this public repository or its issue tracker.

The current code does not load external model files or make network requests. Future file parsers must validate lengths, offsets, integer arithmetic, and resource limits before allocation or access. Imported model files should remain immutable by default. Report suspected vulnerabilities privately to the repository owner rather than publishing exploitable details before a fix is available.

No security contact or supported release process has been configured yet.
