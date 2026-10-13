#!/usr/bin/env python3
"""No-GPU contract test for official infer.py prompt insertion and S0 invariance."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
WORK5 = ROOT / "research/main/Phase2/5.full_benchmark_prompt_study"
EVAL = ROOT / "reproduction/1. Vision-OPD/code/eval/infer.py"
REGISTRY = WORK5 / "prompts/prompt_registry-v1.json"


def main() -> None:
    if "openai" not in sys.modules:
        sys.modules["openai"] = SimpleNamespace(OpenAI=object)
    spec = importlib.util.spec_from_file_location("vision_opd_infer_work5_test", EVAL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.image_to_data_uri = lambda _path, _benchmark: "data:image/png;base64,dGVzdA=="
    seen = {}

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            seen["messages"] = kwargs["messages"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="B"))],
                usage=SimpleNamespace(prompt_tokens=17, completion_tokens=3, total_tokens=20),
            )

    module.OpenAI = FakeClient
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="work5-infer-contract-") as temp:
        root = Path(temp)
        input_path = root / "input.json"
        input_path.write_text(json.dumps([{
            "sample_uid": "toy:1",
            "images": ["unused.jpg"],
            "query": "Original user query; do not modify.",
            "response": "B",
        }]), encoding="utf-8")
        for condition, config in registry["conditions"].items():
            out_dir = root / condition
            sys.argv = [
                str(EVAL), "--benchmark", "toy", "--benchmark_json", str(input_path),
                "--out_dir", str(out_dir), "--model_name", f"contract-{condition}",
                "--api_base", "http://127.0.0.1:1/v1", "--model_id", "fake",
                "--prompt_registry", str(REGISTRY), "--condition", condition,
                "--parallel_workers", "1",
            ]
            module.main()
            messages = seen["messages"]
            expected_roles = ["user"] if condition == "S0" else ["system", "user"]
            assert [message["role"] for message in messages] == expected_roles, condition
            assert messages[-1]["content"][1]["text"] == "Original user query; do not modify."
            if condition != "S0":
                assert messages[0]["content"] == config["system_message"]
            output = out_dir / "toy" / f"contract-{condition}_answer.jsonl"
            record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            assert record["prompt_condition"] == condition
            assert record["model_answer"] == "B"
            assert record["api_usage"] == {"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20}
            assert record["system_prompt_sha256"] is None if condition == "S0" else len(record["system_prompt_sha256"]) == 64
    print("PASS: S0 has one user message; SC/S1/S2/S3 add only the registered system message.")


if __name__ == "__main__":
    main()
