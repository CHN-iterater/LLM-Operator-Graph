# Project conventions

- Write code comments, docstrings, documentation, CLI messages and generated report prose in English.
- Preserve user-supplied prompts, model names and source paths as data. Do not translate historical filenames or change provenance values.
- Keep the local Hugging Face weights-to-JSON-and-reports workflow available through `operator-graph export`.
- Treat full-prompt inputs and independent single-token probes as separate cache-disabled forwards. Do not label the probe as cached decoding.
- Preserve raw operator types, ordered operand uses, tensor configurations and deduplicated call dependencies.
- Keep static arithmetic-work and logical-access estimates distinct from energy measurements. Preserve unavailable values separately from known zeros.
- Use the documented estimation version and verify numerical changes against the paper archives before changing rules.
- Keep optional model/export dependencies out of core JSON analysis imports.
- Verify generated English reports as well as numerical results when changing report text.
- Run `python -m pytest` with the required extras installed. See `docs/migration_report.md` for the tested environment and local Anaconda workaround.
- Keep the source repository and paper artifacts unchanged during work in this independent project.
