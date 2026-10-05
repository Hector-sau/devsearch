"""Serving layer: the shared search pipeline, single-slot admission, the release bundle and the API/UI.

`pipeline.SearchPipeline` is the one place that composes retrieval, reranking and fallback; the CLI
and the HTTP service both call it.
"""
