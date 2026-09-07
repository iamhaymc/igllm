"""Offline tests: python3 -m unittest weights_train_test"""

import json
import tempfile
import unittest
from pathlib import Path

import weights_train as train


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        return [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"gemma4"}')
        (self.model / "model.safetensors").write_bytes(b"\0" * 100)

    def test_example_dataset(self):
        self.assertEqual(len(train.read_examples(train.ROOT / "weights_train.jsonl")), 4)

    def test_invalid_datasets(self):
        path = self.root / "data.jsonl"
        for content in ("", "\n", "{", "[]", '{"prompt":"x"}',
                        '{"prompt":" ","response":"x"}',
                        '{"prompt":2,"response":"x"}'):
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    train.read_examples(path)

    def test_blank_lines_are_ignored(self):
        path = self.root / "data.jsonl"
        path.write_text('\n{"prompt":"p","response":"r"}\n\n')
        self.assertEqual(train.read_examples(path), [{"prompt": "p", "response": "r"}])

    def test_source_is_never_an_output(self):
        for output in (self.model, self.model / "trained", self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                train.check_paths(self.model, output)

    def test_existing_output_is_rejected(self):
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(ValueError):
            train.check_paths(self.model, output)

    def test_valid_paths(self):
        self.assertEqual(train.check_paths(self.model, self.root / "new")["model_type"], "gemma4")

    def test_lfs_pointer(self):
        (self.model / "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(ValueError, "Git LFS"):
            train.check_paths(self.model, self.root / "new")

    def test_missing_shard(self):
        (self.model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "missing.safetensors"}})
        )
        with self.assertRaises(OSError):
            train.check_paths(self.model, self.root / "new")

    def test_shard_path_traversal(self):
        (self.model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "../outside.safetensors"}})
        )
        with self.assertRaisesRegex(ValueError, "escapes"):
            train.check_paths(self.model, self.root / "new")

    def test_prompt_tokens_are_masked(self):
        result = train.tokenize_examples(Tokenizer(), [{"prompt": "p", "response": "r"}], 5)
        self.assertEqual(result[0]["input_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(result[0]["labels"], [-100, -100, -100, 4, 5])

    def test_long_examples_are_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            train.tokenize_examples(Tokenizer(), [{"prompt": "p", "response": "r"}], 4)

    def test_inconsistent_template(self):
        class BadTokenizer(Tokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
                return [1, 2] if add_generation_prompt else [3, 4]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            train.tokenize_examples(BadTokenizer(), [{"prompt": "p", "response": "r"}], 10)

    def test_empty_response_tokenization(self):
        class EmptyTokenizer(Tokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
                return [1, 2]
        with self.assertRaisesRegex(ValueError, "no response"):
            train.tokenize_examples(EmptyTokenizer(), [{"prompt": "p", "response": "r"}], 10)

    def test_dry_run_does_not_create_output(self):
        output = self.root / "new"
        self.assertEqual(train.main(["--model", str(self.model), "--output", str(output),
                                     "--dry-run"]), 0)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
