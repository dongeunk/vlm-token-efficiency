# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dongeunk (original project contributions).
# Licensed under the Apache License, Version 2.0 (the License).
# You may not use this file except in compliance with the License.
# A copy is in the repository LICENSE and at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by law or agreed to in writing, distributed AS IS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for permissions and limitations.
# Publication copy: license/attribution comments added; executable code unchanged.

"""Runner contract tests with test-only fakes and an in-memory filesystem.

No POPE/model outputs or fixture files are written. The tiny fake generation
below tests code plumbing only; it is not a pilot result or model evaluation.
"""

import builtins
from contextlib import contextmanager, ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from examples import run_pope_pilot as pilot
from src.evaluation.pope import sample_balanced


class FakeBatch(dict):
    def to(self, device):
        self.requested_device = device
        return self


class FakeProcessor:
    """A test-only processor that records the unmodified chat and tensor inputs."""

    def __init__(self, inputs=None):
        self.inputs = inputs if inputs is not None else FakeBatch(
            input_ids=torch.tensor([[3, 99, 99, 8, 9]]),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            pixel_values=torch.zeros(1, 1, 3, 2, 2),
        )
        self.messages = None
        self.template_kwargs = None
        self.process_kwargs = None
        self.decode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.template_kwargs = messages, kwargs
        return "<standard-chat>" + messages[0]["content"][1]["text"] + "<assistant>"

    def __call__(self, **kwargs):
        self.process_kwargs = kwargs
        return self.inputs

    def decode(self, tokens, **kwargs):
        self.decode_calls.append((tokens.tolist(), kwargs))
        if kwargs["skip_special_tokens"]:
            return "  Yes, there is a cat.\n"
        return "<test-only-special>  Yes, there is a cat.\n"


class FakeModel:
    """Test-only full generate stub; feature/insertion paths must never be used."""

    def __init__(self, generation=None, wrong_prefix=False):
        self.config = SimpleNamespace(image_token_id=99)
        self.generation = [50, 51] if generation is None else generation
        self.wrong_prefix = wrong_prefix
        self.calls = []

    def generate(self, **kwargs):
        if not torch.is_inference_mode_enabled():
            raise AssertionError("Full generation must run under inference_mode")
        self.calls.append(kwargs)
        prefix = kwargs["input_ids"].clone()
        if self.wrong_prefix:
            prefix[0, 0] += 1
        generated = torch.tensor([self.generation], dtype=torch.long)
        return SimpleNamespace(sequences=torch.cat((prefix, generated), dim=1))

    def get_image_features(self, **kwargs):
        raise AssertionError("Feature extraction/insertion must not run in the full-model pilot")

    def get_input_embeddings(self):
        raise AssertionError("Direct visual insertion must not run in the full-model pilot")


class MemoryFilesystem:
    """Minimal Path I/O backend; every apparent output remains in this dict."""

    def __init__(self, files=None):
        self.files = {str(path): value if isinstance(value, bytes) else value.encode("utf-8")
                      for path, value in (files or {}).items()}
        self.directories = set()

    def bytes_at(self, path):
        key = str(path)
        if key not in self.files:
            raise FileNotFoundError(key)
        return self.files[key]

    def text_at(self, path, encoding="utf-8"):
        return self.bytes_at(path).decode(encoding)

    def write_text(self, path, text, encoding="utf-8", **kwargs):
        self.files[str(path)] = text.encode(encoding)
        return len(text)

    def open(self, path, mode="r", encoding=None, **kwargs):
        if mode != "r":
            raise AssertionError(f"Unexpected filesystem open mode in data-only test: {mode}")
        return io.StringIO(self.text_at(path, encoding or "utf-8"))

    def mkdir(self, path, **kwargs):
        self.directories.add(str(path))

    @contextmanager
    def patched(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "is_file", lambda path: str(path) in self.files))
            stack.enter_context(patch.object(Path, "exists", lambda path: str(path) in self.files or str(path) in self.directories))
            stack.enter_context(patch.object(Path, "read_bytes", lambda path: self.bytes_at(path)))
            stack.enter_context(patch.object(Path, "read_text", lambda path, encoding="utf-8", **kwargs: self.text_at(path, encoding)))
            stack.enter_context(patch.object(Path, "write_text", lambda path, text, encoding="utf-8", **kwargs: self.write_text(path, text, encoding, **kwargs)))
            stack.enter_context(patch.object(Path, "open", lambda path, mode="r", encoding=None, **kwargs: self.open(path, mode, encoding, **kwargs)))
            stack.enter_context(patch.object(Path, "mkdir", lambda path, **kwargs: self.mkdir(path, **kwargs)))
            yield self


def annotation_fixture():
    # Deliberately more balanced/distinct records than the requested pilot size.
    return [dict(question_id=index, image=f"COCO_val2014_{index:012d}.jpg",
                 text=f"Is there a cat in image {index}?", label="yes" if index % 2 else "no")
            for index in range(1, 81)]


class FullGenerationContractTest(unittest.TestCase):
    def test_full_chat_generation_preserves_question_pixels_tokens_and_raw_output(self):
        processor, model = FakeProcessor(), FakeModel()
        image = object()
        question = "  Is there a cat in the image?  "
        result = pilot.generate_answer(model, processor, image, question, torch, device="cpu")
        self.assertEqual(processor.messages, [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}])
        self.assertEqual(processor.template_kwargs, {"add_generation_prompt": True})
        self.assertEqual(processor.process_kwargs, dict(
            text="<standard-chat>" + question + "<assistant>", images=[image], return_tensors="pt"))
        self.assertEqual(processor.inputs.requested_device, "cpu")
        self.assertEqual(len(model.calls), 1)
        call = model.calls[0]
        for name, value in pilot.GENERATION.items():
            self.assertEqual(call[name], value)
        self.assertIs(call["do_sample"], False)
        self.assertEqual(call["max_new_tokens"], 16)
        self.assertEqual(call["num_beams"], 1)
        self.assertIs(call["input_ids"], processor.inputs["input_ids"])
        self.assertIs(call["pixel_values"], processor.inputs["pixel_values"])
        self.assertEqual(call["input_ids"].shape[0], 1)
        self.assertNotIn("inputs_embeds", call)
        self.assertEqual(set(call), set(processor.inputs) | set(pilot.GENERATION))
        self.assertEqual(result["original_visual_tokens"], 2)
        self.assertEqual(result["retained_visual_tokens"], 2)
        self.assertEqual(result["input_sequence_length"], 5)
        self.assertEqual(result["generated_token_ids"], [50, 51])
        self.assertEqual(result["raw_output"], "  Yes, there is a cat.\n")
        self.assertEqual(result["raw_output_with_special_tokens"], "<test-only-special>  Yes, there is a cat.\n")
        self.assertEqual(result["parsed_answer"], "yes")
        self.assertEqual(processor.decode_calls, [
            ([50, 51], dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)),
            ([50, 51], dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)),
        ])
        self.assertGreaterEqual(result["generation_seconds"], 0)
        self.assertIsNone(result["generation_peak_cuda_mib"])

    def test_unexpected_batch_missing_visual_tokens_or_pixels_stops_before_generate(self):
        fixtures = [
            FakeBatch(input_ids=torch.tensor([[99, 1], [99, 2]]), pixel_values=torch.zeros(2, 1, 3, 2, 2)),
            FakeBatch(input_ids=torch.tensor([[3, 1]]), pixel_values=torch.zeros(1, 1, 3, 2, 2)),
            FakeBatch(input_ids=torch.tensor([[99, 1]])),
        ]
        for inputs in fixtures:
            with self.subTest(input_keys=list(inputs)):
                model = FakeModel()
                with self.assertRaisesRegex(RuntimeError, "batch_size=1"):
                    pilot.generate_answer(model, FakeProcessor(inputs), object(), "Is there a cat?", torch, "cpu")
                self.assertEqual(model.calls, [])

    def test_changed_prefix_and_excess_generated_tokens_are_rejected(self):
        for model, message in ((FakeModel(wrong_prefix=True), "prompt token prefix"),
                               (FakeModel(generation=list(range(17))), "token budget")):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                pilot.generate_answer(model, FakeProcessor(), object(), "Is there a cat?", torch, "cpu")


class DataOnlyPilotContractTest(unittest.TestCase):
    def fixture(self):
        root = Path(__file__).resolve().parents[1] / "_test_only_memory_pope"
        args = SimpleNamespace(annotations=root / "annotations.json", image_dir=root / "images",
                               output_dir=root / "results", seed=42, prepare_only=False)
        records = annotation_fixture()
        text = "".join(json.dumps(record) + "\n" for record in records)
        digest = hashlib.sha256(text.encode()).hexdigest()
        fs = MemoryFilesystem({args.annotations: text})
        return args, records, digest, fs

    @contextmanager
    def no_model_imports(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "transformers"):
                raise AssertionError(f"Missing-data path attempted model/package import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with patch.object(pilot, "generate_answer", side_effect=AssertionError("No generation allowed")):
                yield

    def test_missing_selected_images_do_not_trigger_available_image_resampling(self):
        args, records, digest, fs = self.fixture()
        expected = sample_balanced(records, per_label=15, seed=42)
        selected_ids = {record["question_id"] for record in expected}
        # Fifty other distinct images are available, with enough of each class;
        # none of the chosen thirty exist. They must not replace the manifest.
        for record in records:
            if record["question_id"] not in selected_ids:
                fs.files[str(args.image_dir / record["image"])] = b"unused test-only fake bytes"
        config = {"seed": 42, "annotation_source_verified": False}
        with fs.patched(), patch.object(pilot, "ANNOTATION_SHA256", digest):
            manifest, preflight = pilot.prepare(config, args.annotations, args.image_dir)
        self.assertEqual([record["question_id"] for record in manifest], [record["question_id"] for record in expected])
        self.assertEqual(len(manifest), 30)
        self.assertEqual(preflight["selected_label_counts"], {"yes": 15, "no": 15})
        self.assertEqual(preflight["distinct_images"], 30)
        self.assertEqual(len(preflight["missing_images"]), 30)
        self.assertTrue(all(not record["image_exists"] for record in manifest))
        self.assertIs(preflight["data_ready"], False)

    def test_missing_images_block_before_model_imports_and_save_no_fabricated_metrics(self):
        args, records, digest, fs = self.fixture()
        with fs.patched(), patch.object(pilot, "ANNOTATION_SHA256", digest), \
                patch.object(pilot, "package_versions", return_value={"test_only": True}), \
                self.no_model_imports(), redirect_stdout(io.StringIO()):
            code = pilot.run(args)
        self.assertEqual(code, 2)
        summary = json.loads(fs.text_at(args.output_dir / "summary.json"))
        self.assertEqual(summary["status"], "blocked_missing_or_invalid_data")
        self.assertIsNone(summary["metrics"])
        self.assertIs(summary["model_loaded"], False)
        self.assertIs(summary["generation_executed"], False)
        self.assertEqual(summary["completed_questions"], 0)
        self.assertEqual(summary["next_experiment"], "not_started")
        manifest = [json.loads(line) for line in fs.text_at(args.output_dir / "sample_manifest.jsonl").splitlines()]
        self.assertEqual(len(manifest), 30)
        self.assertEqual(len({record["image"] for record in manifest}), 30)
        self.assertEqual(sum(record["label"] == "yes" for record in manifest), 15)
        self.assertNotIn(str(args.output_dir / "predictions.jsonl"), fs.files)

    def test_pinned_annotation_hash_mismatch_stops_before_annotation_selection(self):
        args, _, _, fs = self.fixture()
        with fs.patched(), patch.object(pilot, "ANNOTATION_SHA256", "0" * 64), \
                patch.object(pilot, "load_annotations", side_effect=AssertionError("Do not load mismatched data")):
            manifest, preflight = pilot.prepare({"seed": 42}, args.annotations, args.image_dir)
        self.assertEqual(manifest, [])
        self.assertIs(preflight["data_ready"], False)
        self.assertTrue(any("SHA256" in error for error in preflight["errors"]))

    def test_existing_result_file_prevents_every_output_write(self):
        args, _, digest, fs = self.fixture()
        path = args.output_dir / "summary.json"
        fs.files[str(path)] = b'{"existing": "preserve exactly"}\n'
        before = dict(fs.files)
        with fs.patched(), patch.object(pilot, "ANNOTATION_SHA256", digest), \
                self.no_model_imports(), self.assertRaises(FileExistsError):
            pilot.run(args)
        self.assertEqual(fs.files, before)

    def test_prepare_only_complete_manifest_still_never_loads_model_or_scores(self):
        args, records, digest, fs = self.fixture()
        args.prepare_only = True
        for record in records:
            fs.files[str(args.image_dir / record["image"])] = b"test-only placeholder; never decoded"
        with fs.patched(), patch.object(pilot, "ANNOTATION_SHA256", digest), \
                patch.object(pilot, "package_versions", return_value={"test_only": True}), \
                self.no_model_imports(), redirect_stdout(io.StringIO()):
            code = pilot.run(args)
        self.assertEqual(code, 0)
        summary = json.loads(fs.text_at(args.output_dir / "summary.json"))
        self.assertEqual(summary["status"], "prepared_only")
        self.assertIsNone(summary["metrics"])
        self.assertIs(summary["model_loaded"], False)
        self.assertIs(summary["generation_executed"], False)


if __name__ == "__main__":
    unittest.main()
