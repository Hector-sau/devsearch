"""Static readers and the chunker (T03): extraction is faithful, nothing is executed, nothing is truncated.

Standard library only: a whitespace token counter stands in for the pinned tokenizer, so these prove the structure and
bookkeeping, not real token lengths (the real-tokenizer run is a separate, gated check).
"""

import json
import tempfile
import unittest
from pathlib import Path

from devsearch.data.corpus import DocInput, load_corpus, write_corpus
from devsearch.ingestion.chunker import ChunkingPolicy, chunk_document
from devsearch.ingestion.gallery import parse_gallery
from devsearch.ingestion.markdown import parse_lines, parse_markdown
from devsearch.ingestion.model import CellSpan, ParseFailure
from devsearch.ingestion.notebook import parse_notebook
from devsearch.ingestion.rst import parse_rst


class Words:
    """A stand-in tokenizer: one token per whitespace-separated word."""

    def count(self, text: str) -> int:
        return len(text.split())


def kinds(extraction):
    return [(b.kind, b.level) if b.kind == "heading" else b.kind for b in extraction.blocks]


def texts(extraction):
    return [b.text for b in extraction.blocks]


class MarkdownTests(unittest.TestCase):
    DOC = """---
title: Gradient Guide
draft: false
---

# Gradients

Accumulation sums gradients.

## Usage

Call backward, then step.

```python
# not a heading
loss.backward()
```

Setext Title
------------

<!-- a comment
spanning lines -->

Last paragraph.
"""

    def test_structure_title_front_matter_and_verbatim_lines(self):
        extraction = parse_markdown(self.DOC)
        self.assertEqual(extraction.title, "Gradient Guide")
        self.assertEqual(kinds(extraction), [("heading", 1), "text", ("heading", 2), "text", "code", ("heading", 2), "text"])
        self.assertEqual(texts(extraction)[4], "# not a heading\nloss.backward()")          # the fence content stays code, fence markers go
        self.assertEqual(extraction.dropped["front-matter"], 4)
        self.assertEqual(extraction.dropped["html-comment"], 2)
        self.assertEqual(extraction.dropped["code-fence-marker"], 2)
        numbers = [line.origin for block in extraction.blocks for line in block.lines]
        self.assertEqual(numbers, sorted(numbers))
        source = self.DOC.split("\n")
        for block in extraction.blocks:
            for line in block.lines:
                self.assertEqual(line.text, source[line.origin - 1], "kept lines are verbatim and carry their own line number")

    def test_unterminated_constructs_are_quarantined_not_guessed(self):
        for body, needle in (("text\n\n```python\ncode forever\n", "never closed"), ("<!-- open\nstill open\n", "never closed"), ("---\ntitle: x\n", "never closed")):
            with self.subTest(body=body):
                with self.assertRaisesRegex(ParseFailure, needle):
                    parse_markdown(body)

    def test_tilde_fences_and_longer_closing_fences(self):
        extraction = parse_markdown("~~~\ncode\n~~~\n\n````\n```\nnested\n```\n````\n")
        self.assertEqual(kinds(extraction), ["code", "code"])
        self.assertEqual(texts(extraction)[1], "```\nnested\n```")

    def test_mdx_statements_and_jsx_lines_are_dropped_only_in_mdx(self):
        source = "import Tabs from '@theme/Tabs';\n\n# Title\n\n<Tip>\n\nUse the small batch.\n\n</Tip>\n\n[[autodoc]] BertModel\n    - forward\n\nDone.\n"
        mdx, plain = parse_markdown(source, mdx=True), parse_markdown(source)
        self.assertEqual(mdx.dropped["mdx-statement"], 1)
        self.assertEqual(mdx.dropped["jsx-tag-line"], 2)
        self.assertEqual(mdx.dropped["autodoc-directive"], 2)
        self.assertEqual(texts(mdx), ["# Title", "Use the small batch.", "Done."])
        self.assertIn("<Tip>", " ".join(texts(plain)))

    def test_binary_overlong_and_nul_content_is_refused(self):
        with self.assertRaisesRegex(ParseFailure, "NUL"):
            parse_markdown("abc\x00def")
        with self.assertRaisesRegex(ParseFailure, "characters"):
            parse_markdown("x" * 10_001)

    def test_cell_mode_gives_exact_spans_into_the_cell_text(self):
        cell = "# Title\n\nSome words here.\n\n```\ncode()\n```"
        extraction = parse_lines(cell.split("\n"), cell_id="c1")
        for block in extraction.blocks:
            for line in block.lines:
                self.assertIsInstance(line.origin, CellSpan)
                self.assertEqual(cell[line.origin.start:line.origin.end], line.text)


class RstTests(unittest.TestCase):
    DOC = """.. _intro:

Training a Model
================

This shows :func:`torch.nn.Module` and :class:`~torch.optim.SGD`; see ``loss`` and `Docs <https://x.y>`_.

.. note::

   Call :meth:`zero_grad` first.

   Example::

       x = 1

.. toctree::
   :maxdepth: 2

   other

.. image:: pic.png

Example::

    import torch
    x = torch.zeros(3)

Section Two
-----------

.. code-block:: python
   :linenos:

   y = x + 1
   print(y)

.. unknowndirective:: arg

   hidden body line

.. a comment
   with a body

+-----+-----+
| a   | b   |
+-----+-----+

Done.
"""

    def test_headings_roles_literals_and_directives(self):
        extraction = parse_rst(self.DOC)
        self.assertEqual(extraction.title, "Training a Model")
        self.assertEqual(kinds(extraction), [("heading", 1), "text", "text", "text", "code", "text", "code", ("heading", 2), "code", "text", "text"])
        flat = "\n".join(texts(extraction))
        self.assertIn("This shows torch.nn.Module and torch.optim.SGD; see loss and Docs.", flat)
        self.assertIn("Call zero_grad first.", flat)
        self.assertNotIn(":func:", flat)
        self.assertNotIn("hidden body line", flat)
        self.assertNotIn("other", flat.split())
        code = [b.text for b in extraction.blocks if b.kind == "code"]
        self.assertEqual(code, ["x = 1", "import torch\nx = torch.zeros(3)", "y = x + 1\nprint(y)"])
        self.assertEqual(extraction.dropped["rst-directive:toctree"], 4)
        self.assertEqual(extraction.dropped["rst-directive:image"], 1)
        self.assertEqual(extraction.dropped["rst-directive:unknowndirective"], 3)
        self.assertEqual(extraction.dropped["rst-target"], 1)
        self.assertEqual(extraction.dropped["rst-comment"], 2)
        self.assertEqual(extraction.dropped["rst-table-border"], 2)

    def test_lines_keep_their_source_line_numbers(self):
        source = self.DOC.split("\n")
        extraction = parse_rst(self.DOC)
        code = next(b for b in extraction.blocks if b.text.startswith("import torch"))
        self.assertEqual([source[line.origin - 1].strip() for line in code.lines], ["import torch", "x = torch.zeros(3)"])

    def test_heading_levels_follow_first_appearance(self):
        extraction = parse_rst("=====\nTitle\n=====\n\nSub\n---\n\ntext\n\nDeep\n~~~~\n\nmore\n\nSub two\n-------\n\nend\n")
        self.assertEqual([(b.level, b.title) for b in extraction.blocks if b.kind == "heading"], [(1, "Title"), (2, "Sub"), (3, "Deep"), (2, "Sub two")])

    def test_an_underline_shorter_than_its_title_is_still_a_heading_but_a_tiny_adornment_is_not(self):
        # docutils only warns about a short underline, so a title such as the gallery docstring of torch_compile_full_example.py is a heading
        extraction = parse_rst("``torch.compile`` End-to-End Tutorial\n=================================\n\nintro text\n\nOverview of the details\n------\n\nbody\n")
        self.assertEqual([(b.level, b.title) for b in extraction.blocks if b.kind == "heading"], [(1, "torch.compile End-to-End Tutorial"), (2, "Overview of the details")])
        self.assertEqual(extraction.title, "torch.compile End-to-End Tutorial")
        # three characters under a long line is not an underline (docutils needs four for a short one), so it stays text
        tiny = parse_rst("A line that is much longer than its marker\n---\n\nbody\n")
        self.assertEqual([b.title for b in tiny.blocks if b.kind == "heading"], [])

    def test_layout_containers_are_transparent_and_other_directive_bodies_are_not(self):
        source = """.. tab-set::

   .. tab-item:: PyTorch

      Use the module like this.

      .. code-block:: python

         y = model(x)

.. dropdown:: Details
   :open:

   Hidden but real explanation text.

.. grid:: 2

   .. grid-item-card:: Card
      :link: x

      Card text appears.

.. math::

   a^2 + b^2 = c^2

.. list-table::

   * - cell
"""
        extraction = parse_rst(source)
        flat = "\n".join(texts(extraction))
        for expected in ("Use the module like this.", "y = model(x)", "Hidden but real explanation text.", "Card text appears."):
            self.assertIn(expected, flat)
        for absent in ("a^2", "cell", ":open:", ":link:"):
            self.assertNotIn(absent, flat)
        self.assertEqual(extraction.dropped["rst-directive:math"], 3)
        self.assertEqual(extraction.dropped["rst-directive:list-table"], 3)

    def test_markup_that_spans_lines_is_reduced(self):
        extraction = parse_rst("See `Process Group\nInitialization <https://example.org/x#y>`__ and :class:`torch.nn.\nModule` for more.\n")
        self.assertEqual(" ".join(texts(extraction)).split(), "See Process Group Initialization and torch.nn. Module for more.".split())
        self.assertNotIn("example.org", " ".join(texts(extraction)))

    def test_headings_become_clean_one_line_labels(self):
        extraction = parse_rst("A\u00a0title\twith   odd\u200bspacing\n========================================\n\ntext\n")
        self.assertEqual(extraction.title, "A title with odd spacing")
        long = parse_markdown("# " + "word " * 100 + "\n\nbody\n")
        self.assertLessEqual(len(long.title), 200)

    def test_escapes_and_a_bare_double_colon(self):
        extraction = parse_rst("Plain \\*not emphasis\\* text\n\n::\n\n    raw block\n")
        self.assertEqual(texts(extraction), ["Plain *not emphasis* text", "raw block"])
        self.assertEqual(kinds(extraction), ["text", "code"])


class GalleryTests(unittest.TestCase):
    SOURCE = '''# Copyright notice that is boilerplate
"""
Training basics
===============

Short introduction to **training**.
"""

import torch

######################################################################
# First step
# ----------
#
# We create a tensor and look at it.

x = torch.zeros(3)
print(x)

# %%
# A note without a heading.

y = x + 1  # trailing comment
'''

    def test_docstring_comment_prose_and_code_blocks(self):
        extraction = parse_gallery(self.SOURCE)
        self.assertEqual(extraction.title, "Training basics")
        self.assertEqual(kinds(extraction), [("heading", 1), "text", "code", ("heading", 2), "text", "code", "text", "code"])
        self.assertEqual(extraction.blocks[2].text, "import torch")
        self.assertEqual(extraction.blocks[5].text, "x = torch.zeros(3)\nprint(x)")
        self.assertEqual(extraction.dropped["gallery-separator"], 2)
        numbers = [line.origin for block in extraction.blocks for line in block.lines]
        self.assertEqual(numbers, sorted(numbers))
        source = self.SOURCE.split("\n")
        self.assertEqual(source[extraction.blocks[5].lines[0].origin - 1], "x = torch.zeros(3)")

    def test_the_file_is_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            hostile = f'"""\nTitle\n=====\n"""\nopen({str(marker)!r}, "w").write("x")\nraise SystemExit(7)\n'
            extraction = parse_gallery(hostile)
            self.assertFalse(marker.exists())
            self.assertEqual(kinds(extraction), [("heading", 1), "code"])

    def test_invalid_python_is_quarantined(self):
        with self.assertRaisesRegex(ParseFailure, "not valid Python"):
            parse_gallery("def broken(:\n    pass\n")

    def test_a_docstring_with_escapes_or_continuations_is_read_raw_and_noted(self):
        source = '"""\nTitle\n=====\n\nA line with a continuation \\\nthat joins here.\n"""\nx = 1\n'
        extraction = parse_gallery(source)
        self.assertEqual(extraction.notes["docstring-read-raw"], 1)
        self.assertEqual(extraction.title, "Title")
        self.assertIn("that joins here.", "\n".join(texts(extraction)))
        one_line = parse_gallery('"""Caf\u00e9 doc \\\ncontinued"""\ny = 2\n')
        self.assertIn("Caf\u00e9 doc", texts(one_line)[0], "non-ASCII text keeps its columns")

    def test_a_file_without_a_docstring_is_recorded(self):
        extraction = parse_gallery("import os\n\nx = 1\n")
        self.assertEqual(extraction.dropped["no-module-docstring"], 1)
        self.assertEqual(kinds(extraction), ["code"])


class NotebookTests(unittest.TestCase):
    def notebook(self, cells, **top):
        return json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells, **top})

    def test_cells_become_blocks_with_exact_cell_spans_and_outputs_are_ignored(self):
        cells = [
            {"cell_type": "markdown", "id": "intro", "metadata": {}, "source": ["# Tensors\n", "\n", "Tensors hold numbers.\n"]},
            {"cell_type": "code", "id": "c1", "metadata": {}, "source": "x = torch.zeros(2)\nprint(x)\n", "outputs": [{"output_type": "stream", "text": "SECRET OUTPUT"}], "execution_count": 1},
            {"cell_type": "raw", "id": "r", "metadata": {}, "source": "raw text"},
            {"cell_type": "code", "metadata": {}, "source": "", "outputs": []},
        ]
        extraction = parse_notebook(self.notebook(cells))
        self.assertEqual(extraction.title, "Tensors")
        self.assertEqual(kinds(extraction), [("heading", 1), "text", "code"])
        self.assertNotIn("SECRET OUTPUT", " ".join(texts(extraction)))
        self.assertEqual(extraction.notes["code-cell-outputs-ignored"], 1)
        self.assertEqual(extraction.dropped["raw-cell"], 1)
        by_id = {"intro": "# Tensors\n\nTensors hold numbers.\n", "c1": "x = torch.zeros(2)\nprint(x)\n"}
        for block in extraction.blocks:
            for line in block.lines:
                self.assertEqual(by_id[line.origin.cell_id][line.origin.start:line.origin.end], line.text)

    def test_bad_notebooks_are_quarantined_with_a_reason(self):
        cases = {
            "not json": "{nope",
            "no cells": json.dumps({"nbformat": 4}),
            "old format": json.dumps({"nbformat": 3, "cells": []}),
            "nan": '{"nbformat": 4, "cells": [], "metadata": {"x": NaN}}',
            "bad cell type": self.notebook([{"cell_type": "weird", "source": ""}]),
            "bad source": self.notebook([{"cell_type": "code", "source": 5}]),
        }
        for label, text in cases.items():
            with self.subTest(label):
                with self.assertRaises(ParseFailure):
                    parse_notebook(text)

    def test_duplicate_or_invalid_cell_ids_fall_back_to_the_index(self):
        cells = [{"cell_type": "code", "id": "same", "source": "a = 1"}, {"cell_type": "code", "id": "same", "source": "b = 2"},
                 {"cell_type": "code", "id": "bad id!", "source": "c = 3"}]
        ids = [block.lines[0].origin.cell_id for block in parse_notebook(self.notebook(cells)).blocks]
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(ids[0], "same")


class ChunkerTests(unittest.TestCase):
    DOC = """# Alpha

Intro words that explain the alpha module in a handful of plain words.

## Beta

First paragraph about beta with some words in it today.

Second paragraph about beta with a few more words in it too.

```python
one = 1
two = 2
three = 3
```

## Gamma

Gamma sentence one is here. Gamma sentence two follows it. Gamma sentence three ends it.
"""

    def chunk(self, text=None, budget=16, **policy):
        extraction = parse_markdown(text or self.DOC)
        return chunk_document(extraction, doc_id="d:x.md", revision="r1", path="x.md", counter=Words(), policy=ChunkingPolicy(budget, **policy))

    def assert_sound(self, result, budget):
        for span, tokens in zip(result.chunks, result.tokens):
            text = result.text[span.start_char:span.end_char]
            self.assertEqual(tokens, len(text.split()))
            self.assertLessEqual(tokens, budget, "no passage is ever longer than the budget (and none is truncated)")
            self.assertTrue(text.strip())

    def test_passages_are_whole_blocks_within_the_budget_with_section_groups(self):
        result = self.chunk(budget=20)
        self.assert_sound(result, 20)
        paths = [span.heading_path for span in result.chunks]
        self.assertEqual(paths[0], ("Alpha",))
        self.assertIn(("Alpha", "Beta"), paths)
        self.assertEqual(paths[-1], ("Alpha", "Gamma"))
        groups = {}
        for span in result.chunks:
            groups.setdefault(span.heading_path, set()).add(span.evidence_group_id)
        self.assertTrue(all(len(g) == 1 for g in groups.values()), "one evidence group per section")
        self.assertEqual(len({next(iter(g)) for g in groups.values()}), len(groups), "different sections never share a group")
        first = result.chunks[0]
        self.assertTrue(result.text[first.start_char:first.end_char].startswith("# Alpha"), "a section's first passage starts with its heading")

    def test_overlong_code_is_split_at_lines_and_marked_as_a_fragment(self):
        code = "# T\n\n```python\n" + "\n".join(f"name{i} = {i}" for i in range(8)) + "\n```\n"
        result = self.chunk(text=code, budget=9, min_chunk_tokens=3)
        self.assert_sound(result, 9)
        fragments = [s for s in result.chunks if s.code_fragment]
        self.assertGreaterEqual(len(fragments), 2)
        covered = "\n".join(result.text[s.start_char:s.end_char] for s in result.chunks)
        for i in range(8):
            self.assertEqual(covered.count(f"name{i} = {i}"), 1, "every line lands in exactly one fragment")
        for span in fragments:
            self.assertTrue(all(line.startswith("name") for line in result.text[span.start_char:span.end_char].split("\n") if line and line != "# T"))
        whole = self.chunk(text=code, budget=60, min_chunk_tokens=3)
        self.assertFalse(any(span.code_fragment for span in whole.chunks), "a block that fits is never marked as a fragment")

    def test_the_exclude_policy_drops_an_overlong_code_block_and_counts_it(self):
        result = self.chunk(text="# T\n\n```\n" + "\n".join(f"word{i} = {i}" for i in range(20)) + "\n```\n\nShort closing paragraph for the section here.\n", budget=12, overlong_code="exclude")
        self.assertEqual(result.excluded["overlong-code-block"], 20)
        self.assert_sound(result, 12)
        self.assertFalse(any("word1 =" in result.text[s.start_char:s.end_char] for s in result.chunks))

    def test_prose_is_split_at_sentences_and_a_giant_sentence_is_excluded(self):
        text = "# P\n\n" + "Short sentence number one. Short sentence number two. Short sentence number three.\n"
        result = self.chunk(text=text, budget=8, min_chunk_tokens=3)
        self.assert_sound(result, 8)
        self.assertEqual(len(result.chunks), 2)
        joined = " ".join(result.text[s.start_char:s.end_char] for s in result.chunks)
        for word in ("one.", "two.", "three."):
            self.assertEqual(joined.count(word), 1)
        giant = self.chunk(text="# P\n\n" + "unbroken " * 50 + "\n", budget=8, min_chunk_tokens=3)
        self.assertEqual(giant.excluded["overlong-sentence"], 1)
        self.assertEqual(giant.chunks, ())

    def test_short_passages_and_headings_without_room_are_reported(self):
        short = self.chunk(text="# A\n\nTiny.\n\n# B\n\nA body that is long enough to keep around here.\n", budget=20, min_chunk_tokens=5)
        self.assertEqual(short.excluded["short-passage"], 1)
        self.assertEqual(len(short.chunks), 1)

    def test_chunking_is_deterministic_and_round_trips_through_the_corpus_writer(self):
        a, b = self.chunk(budget=14), self.chunk(budget=14)
        self.assertEqual((a.text, a.chunks, a.segments), (b.text, b.chunks, b.segments))
        with tempfile.TemporaryDirectory() as directory:
            document = DocInput("d:x.md", "src", "T", "r1", "https://example.org/x", "MIT", "static-extract-v1/markdown", a.text, a.segments, a.chunks)
            manifest = write_corpus(Path(directory), [document], chunker_version="chunker-v1")
            corpus = load_corpus(Path(directory) / "corpus-manifest.json")
            self.assertEqual(len(corpus.chunks), len(a.chunks))
            for chunk in corpus.chunks:
                self.assertTrue(corpus.locate(chunk), "every passage maps back to source lines")
            self.assertEqual(manifest["corpus_id"], corpus.corpus_id)

    def test_cell_form_documents_chunk_and_locate_to_cells(self):
        cells = [{"cell_type": "markdown", "id": "m", "source": "# Heading\n\nWords in the markdown cell explain the code that follows it."},
                 {"cell_type": "code", "id": "k", "source": "value = compute()\nprint(value)"}]
        extraction = parse_notebook(json.dumps({"nbformat": 4, "metadata": {}, "cells": cells}))
        result = chunk_document(extraction, doc_id="d:n.ipynb", revision="r1", path="n.ipynb", counter=Words(), policy=ChunkingPolicy(40, min_chunk_tokens=3))
        self.assertTrue(result.chunks)
        with tempfile.TemporaryDirectory() as directory:
            document = DocInput("d:n.ipynb", "src", "N", "r1", None, "MIT", "static-extract-v1/notebook", result.text, result.segments, result.chunks)
            write_corpus(Path(directory), [document], chunker_version="chunker-v1")
            corpus = load_corpus(Path(directory) / "corpus-manifest.json")
            locations = corpus.locate(corpus.chunks[0])
            self.assertTrue(all("cell_id" in location for location in locations), locations)

    def test_an_invalid_policy_is_rejected(self):
        for bad in (dict(passage_budget=4), dict(passage_budget=20, min_chunk_tokens=20), dict(passage_budget=20, overlong_code="truncate")):
            with self.subTest(bad):
                with self.assertRaises(ValueError):
                    ChunkingPolicy(**bad)


if __name__ == "__main__":
    unittest.main()
