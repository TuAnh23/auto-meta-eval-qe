# Code taken from https://github.com/google-research/metricx/blob/main/metricx23/predict.py
# with minor changes for convenient
# coding=utf-8
# Copyright 2024 Google LLC
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
"""Runs inference with a MetricX model."""

from datasets import Dataset
import metricx23_models
import torch
import transformers


def get_dataset(
        input_samples: list, tokenizer, max_input_length: int, device, is_qe: bool
):
    """Gets the test dataset for prediction.

    If `is_qe` is true, the input data must have "hypothesis" and "source" fields.
    If it is false, there must be "hypothesis" and "reference" fields.

    Args:
      input_samples: In-memory input samples
      tokenizer: The tokenizer to use.
      max_input_length: The maximum input sequence length.
      device: The ID of the device to put the PyTorch tensors on.
      is_qe: Indicates whether the metric is a QE metric or not.

    Returns:
      The dataset.
    """

    def _make_input(example):
        if is_qe:
            example["input"] = (
                    "candidate: "
                    + example["hypothesis"]
                    + " source: "
                    + example["source"]
            )
        else:
            example["input"] = (
                    "candidate: "
                    + example["hypothesis"]
                    + " reference: "
                    + example["reference"]
            )
        return example

    def _tokenize(example):
        return tokenizer(
            example["input"],
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )

    def _remove_eos(example):
        example["input_ids"] = example["input_ids"][:-1]
        example["attention_mask"] = example["attention_mask"][:-1]
        return example

    ds = Dataset.from_list(input_samples)
    ds = ds.map(_make_input)
    ds = ds.map(_tokenize)
    ds = ds.map(_remove_eos)
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        device=device,
        output_all_columns=True,
    )
    return ds


def calculate_ref_based_score(mt, ref,
                              tokenizer='google/mt5-xl',
                              model_name_or_path='google/metricx-23-xl-v2p0',
                              max_input_length=1024,
                              batch_size=1):
    """
    :param mt: list of system output sentences
    :param ref: list of reference sentences
    :param tokenizer:
    :param model_name_or_path:
    :param max_input_length:
    :param batch_size:
    :return: list of scores
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        per_device_batch_size = batch_size // torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        per_device_batch_size = batch_size

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)

    model = metricx23_models.MT5ForRegression.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    input_samples = [{"reference": x, "hypothesis": y} for x, y in zip(ref, mt)]
    ds = get_dataset(
        input_samples,
        tokenizer,
        max_input_length,
        device,
        is_qe=False,
    )

    training_args = transformers.TrainingArguments(
        output_dir="tmp_dir_huggingface",
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_pin_memory=False,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
    )
    predictions, _, _ = trainer.predict(test_dataset=ds)
    predictions = [float(x) for x in predictions]

    return predictions
