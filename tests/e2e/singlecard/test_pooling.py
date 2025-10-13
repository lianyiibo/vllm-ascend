#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
import os

import pytest
import torch
from modelscope import snapshot_download  # type: ignore[import-untyped]
from transformers import AutoModelForSequenceClassification

from tests.e2e.conftest import HfRunner, VllmRunner
from tests.e2e.utils import check_embeddings_close

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

EMBED_MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",  # lasttoken
    "BAAI/bge-small-en-v1.5",  # cls_token
    "intfloat/multilingual-e5-small"  # mean_tokens
]
CLASSIFY_MODELS = ["Howeee/Qwen2.5-1.5B-apeach"]
SCORE_MODELS = ["BAAI/bge-reranker-v2-m3"]


@pytest.mark.parametrize("model", EMBED_MODELS)
def test_embed_correctness(model: str) -> None:
    queries = ['What is the capital of China?', 'Explain gravity']

    model_name = snapshot_download(model)
    with VllmRunner(
            model_name,
            runner="pooling",
            enforce_eager=True,
            max_model_len=512,
    ) as vllm_runner:
        vllm_outputs = vllm_runner.embed(queries)

    with HfRunner(
            model_name,
            dtype="float32",
            is_sentence_transformer=True,
    ) as hf_runner:
        hf_outputs = hf_runner.encode(queries)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )


@pytest.mark.parametrize("model", CLASSIFY_MODELS)
def test_classify_correctness(model: str) -> None:

    model_name = snapshot_download(model)
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is what",
    ]
    with VllmRunner(
            model_name,
            runner="pooling",
            enforce_eager=True,
            max_model_len=512,
    ) as vllm_runner:
        vllm_outputs = vllm_runner.classify(prompts)

    with HfRunner(model_name,
                  dtype="float32",
                  auto_cls=AutoModelForSequenceClassification) as hf_runner:
        hf_outputs = hf_runner.classify(prompts)

    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = torch.tensor(hf_output)
        vllm_output = torch.tensor(vllm_output)
        assert torch.allclose(hf_output, vllm_output, 1e-2)


@pytest.mark.parametrize("model", SCORE_MODELS)
def test_score_correctness(model: str) -> None:
    model_name = snapshot_download(model)

    question = "What is the capital of France?"
    options = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
    ]
    text_pairs = [
        [question, options[0]],
        [question, options[1]],
    ]

    with VllmRunner(
            model_name,
            runner="pooling",
            enforce_eager=True,
            max_model_len=None,
    ) as vllm_runner:
        vllm_outputs = vllm_runner.score(question, options)

    with HfRunner(model_name, dtype="half",
                  is_cross_encoder=True) as hf_runner:
        hf_outputs = hf_runner.predict(text_pairs).tolist()

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)
