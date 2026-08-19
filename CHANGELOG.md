# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `latex_math_to_speech` now speaks subscripts: `x_i` reads as "x sub i"
  instead of being emitted verbatim. `_` is tokenised as a subscript marker
  rather than swallowed into the preceding identifier.
- Big operators (`\sum`, `\prod`, `\int`) now handle unbraced limits
  (`\int_0^\infty`) and a missing lower or upper limit. The previous
  three-pattern approach raised `IndexError: no such group` on
  `\sum^{n}` and mis-read `\int_0^\infty` as "integral sub 0 to the power".
- Escaped dollar signs are left alone: `Price is \$5 and equation is $x^2$`
  now converts only the real math.
- `process_latex_document` extracts the abstract into `metadata`.
- `extract_paper_metadata` keeps the text inside formatting commands:
  `\title{A Paper with \textbf{Bold}}` reads as "A Paper with Bold" instead of
  dropping the braced content.
- Figure and table captions are matched to the block above *or* below them,
  so a caption preceding its table is no longer discarded.

### Changed

- Replaced the deprecated, unmaintained `PyPDF2` (final release 3.0.1, subject
  to PYSEC-2026-1835) with its maintained successor `pypdf`.
- Library modules log through the `logging` module instead of printing; the
  CLI still prints, as its output is the user interface.
- arXiv source tarballs are extracted with `filter="data"`, rejecting archive
  entries that would write outside the extraction directory.
- Adopted the [py-canon](https://github.com/gojiplus/py-canon) fleet standard:
  `uv_build` backend, `src/` layout, ruff lint and format, pyright, shared
  reusable CI/docs/release workflows, and PyPI trusted publishing.
- Minimum supported Python is now 3.11.

## [0.3.0]

- Initial released version tracked in this changelog.

[Unreleased]: https://github.com/gojiplus/paper_voice/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/gojiplus/paper_voice/releases/tag/v0.3.0
