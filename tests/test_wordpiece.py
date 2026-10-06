"""The WordPiece pair tokenizer: a `vocab.txt` directory is encoded exactly as its own configuration declares, or refused.

The reference for "exactly" is the independent pure-Python BERT tokenizer that Transformers still ships
(`BertTokenizerLegacy`); it shares no code with the `tokenizers` library the project uses.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import REPO

from mltest import HAVE_ML

if HAVE_ML:
    from devsearch.errors import ManifestError
    from devsearch.modeling.tokenization import WordPieceTokenizer

    try:
        from transformers.models.bert.tokenization_bert_legacy import BertTokenizerLegacy
    except ImportError:                                  # the reference moved or was removed in another Transformers release
        BertTokenizerLegacy = None

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "how", "do", "i", "zero", "the", "grad", "##s", "##ients", "cafe", "café", "Café", "CUDA",
         "cuda", "?", ".", "(", ")", "_", "optimizer", "call", "before", "backward", "中", "文", "中文", "-", "max", "##pool", "2", "##d"]
TEXTS = ["How do I zero the gradients?", "Call optimizer.zero_grad() before backward().", "Café CUDA cafe", "中文 max-pool2d", "  odd\tspacing here  ",
         "unknown-words abcdefghijklmnopqrstuvwxyz", "CAFÉ", ""]


@unittest.skipUnless(HAVE_ML, "torch/transformers/tokenizers/safetensors are not installed")
class VocabDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="devsearch-wordpiece-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        (self.dir / "vocab.txt").write_text("\n".join(VOCAB) + "\n", encoding="utf-8")

    def configure(self, config=None, special_map=None, added=None):
        for name, value in (("tokenizer_config.json", config), ("special_tokens_map.json", special_map), ("added_tokens.json", added)):
            path = self.dir / name
            if value is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(json.dumps(value), encoding="utf-8")
        return WordPieceTokenizer(self.dir)

    def ids(self, tokenizer, text):
        return list(tokenizer._tokenizer.encode(text, add_special_tokens=False).ids)

    def test_every_supported_setting_matches_the_reference_implementation(self):
        if BertTokenizerLegacy is None:
            self.skipTest("the pure-Python reference tokenizer is not available in this Transformers release")
        variants = [{}, {"do_lower_case": True}, {"do_lower_case": False}, {"do_lower_case": True, "strip_accents": False},
                    {"do_lower_case": False, "strip_accents": True}, {"tokenize_chinese_chars": False}]
        for config in variants:
            ours = self.configure(config)
            reference = BertTokenizerLegacy(vocab_file=str(self.dir / "vocab.txt"), do_lower_case=config.get("do_lower_case", True),
                                            strip_accents=config.get("strip_accents"), tokenize_chinese_chars=config.get("tokenize_chinese_chars", True))
            for text in TEXTS:
                self.assertEqual(self.ids(ours, text), reference.convert_tokens_to_ids(reference.tokenize(text)), (config, text))
            query, passage = TEXTS[0], TEXTS[1]
            pair = ours.encode_pair(query, passage)
            q, p = (reference.convert_tokens_to_ids(reference.tokenize(t)) for t in (query, passage))
            self.assertEqual(list(pair.ids), reference.build_inputs_with_special_tokens(q, p), config)
            self.assertEqual(list(pair.type_ids), reference.create_token_type_ids_from_sequences(q, p), config)

    def test_settings_change_the_encoding_and_the_identity(self):
        default = self.configure()
        cased = self.configure({"do_lower_case": False})
        self.assertNotEqual(self.ids(default, "Café CUDA"), self.ids(cased, "Café CUDA"))
        self.assertNotEqual(default.fingerprint, cased.fingerprint)
        accents = self.configure({"do_lower_case": True, "strip_accents": False})
        self.assertNotEqual(self.ids(default, "Café"), self.ids(accents, "Café"), "lowercase strips accents unless strip_accents is false")
        plain = self.configure(special_map={"unk_token": "[UNK]"})
        self.assertNotEqual(plain.fingerprint, default.fingerprint, "special_tokens_map.json is part of the identity")

    def test_settings_the_pipeline_cannot_reproduce_are_refused(self):
        refused = [({"do_basic_tokenize": False}, None, None), ({"never_split": ["CUDA"]}, None, None), ({"unk_token": "<unk>"}, None, None),
                   (None, {"cls_token": {"content": "<s>"}}, None), (None, {"additional_special_tokens": ["[X]"]}, None), (None, None, {"[NEW]": 35}),
                   ({"do_lower_case": "yes"}, None, None), ({"strip_accents": "no"}, None, None),
                   ({"added_tokens_decoder": {"0": {"content": "[PAD]"}, "40": {"content": "[NEW]"}}}, None, None)]
        for config, special_map, added in refused:
            with self.assertRaises(ManifestError, msg=(config, special_map, added)):
                self.configure(config, special_map, added)

    def test_ordinary_hub_configurations_are_accepted(self):
        tokenizer = self.configure({"do_lower_case": True, "model_max_length": 512, "tokenizer_class": "BertTokenizer", "unk_token": "[UNK]",
                                    "added_tokens_decoder": {"0": {"content": "[PAD]"}, "1": {"content": "[UNK]"}}},
                                   {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"}, {})
        self.assertEqual(tokenizer.pair_special_tokens, 3)


@unittest.skipUnless(HAVE_ML, "torch/transformers/tokenizers/safetensors are not installed")
class CompareScriptTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="devsearch-compare-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        for name in ("a", "b"):
            (self.dir / name).mkdir()
            (self.dir / name / "vocab.txt").write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
        self.texts = self.dir / "texts.jsonl"
        self.texts.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS[:-1]), encoding="utf-8")

    def run_compare(self):
        result = subprocess.run([sys.executable, str(REPO / "scripts" / "compare_tokenizers.py"), "--candidate", str(self.dir / "a"), "--reference", str(self.dir / "b"),
                                 "--texts", str(self.texts), "--output", str(self.dir / "report.json")], capture_output=True, text=True, timeout=300)
        return result.returncode, json.loads(result.stdout)

    def test_identical_directories_agree_with_each_other_and_with_the_independent_reference(self):
        code, summary = self.run_compare()
        self.assertEqual(code, 0, summary)
        if BertTokenizerLegacy is not None:
            self.assertIn("bert-legacy", summary["verdict"]["independent_references"])
        self.assertTrue(all(c["identical"] for c in summary["comparisons"].values()))
        self.assertEqual(summary["config_differences"], [])

    def test_a_configuration_difference_is_reported_even_where_the_vocabularies_are_equal(self):
        (self.dir / "b" / "tokenizer_config.json").write_text(json.dumps({"do_lower_case": False}), encoding="utf-8")
        code, summary = self.run_compare()
        self.assertEqual(code, 1)
        self.assertTrue(any("normalizer" in d for d in summary["config_differences"]), summary["config_differences"])
        self.assertGreater(summary["comparisons"]["project-reference"]["texts_differing"], 0)
        report = json.loads((self.dir / "report.json").read_text())
        self.assertEqual(report["candidate"]["effective"]["vocab_sha256"], report["reference"]["effective"]["vocab_sha256"], "same vocab.txt, different encoding")


if __name__ == "__main__":
    unittest.main()
