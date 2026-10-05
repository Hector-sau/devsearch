"""Static source extraction (T03, docs/02 §3, docs/04): pinned snapshots in, validated corpus artifact out.

Nothing here executes, imports or evaluates source content: documents are parsed as text by the readers in this package
(Markdown/MDX, reStructuredText, notebook JSON and sphinx-gallery tutorial Python), turned into structural blocks with exact
source mappings, and cut into passages that are validated against the pinned tokenizer's budget. Unsupported syntax is
dropped line by line (counted, never indexed) or the whole file is quarantined with a reason and a source link.
"""
