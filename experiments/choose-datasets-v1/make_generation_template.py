# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Write Nemotron 3 Nano's chat template with ``{% generation %}`` markers around every assistant turn.

The stock template drops the reasoning of assistant turns that precede the last user turn. ``ChatDataset``'s
fallback loss mask renders conversation prefixes and requires each to be a prefix of the full render, so it
rejects multi-turn conversations with reasoning (most reasoning-on rows of Nemotron-Instruction-Following-Chat-v1).
With the markers, the tokenizer returns the assistant mask directly. The markers wrap each assistant turn exactly
where the fallback mask would (role header through the trailing newline) and do not change the rendered text; the
script checks both on the sampled conversations.

Usage::

    source experiments/derisk-v1/env.sh
    python experiments/choose-datasets-v1/make_generation_template.py \
        --output experiments/choose-datasets-v1/templates/nano3_generation.jinja
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import _normalize_messages
from nemo_automodel.components.datasets.llm.formatting_utils import _build_multiturn_assistant_mask

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# (stock text, tagged text); each stock text must occur exactly once.
_EDITS = (
    (  # assistant turn with tool calls: opens here ...
        "            {{- '<|im_start|>assistant\\n' }}\n                {%- set include_content",
        "            {%- generation %}{{- '<|im_start|>assistant\\n' }}\n                {%- set include_content",
    ),
    (  # ... and closes after its <|im_end|>
        "                {{- '<|im_end|>\\n' }}\n        {%- else %}\n            {# Assistant message doesn't have",
        "                {{- '<|im_end|>\\n' }}{%- endgeneration %}\n        {%- else %}\n"
        "            {# Assistant message doesn't have",
    ),
    (  # last turn(s), reasoning kept
        "{{- '<|im_start|>assistant\\n' ~ (content | default('', true) | string | trim) ~ '<|im_end|>\\n' }}",
        "{%- generation %}{{- '<|im_start|>assistant\\n' ~ (content | default('', true) | string | trim)"
        " ~ '<|im_end|>\\n' }}{%- endgeneration %}",
    ),
    (  # earlier turns, reasoning dropped
        "{{- '<|im_start|>assistant\\n' ~ c ~ '<|im_end|>\\n' }}",
        "{%- generation %}{{- '<|im_start|>assistant\\n' ~ c ~ '<|im_end|>\\n' }}{%- endgeneration %}",
    ),
    (  # earlier turns that are empty after dropping the reasoning
        "{{- '<|im_start|>assistant\\n<|im_end|>\\n' }}",
        "{%- generation %}{{- '<|im_start|>assistant\\n<|im_end|>\\n' }}{%- endgeneration %}",
    ),
)


def tag_template(stock: str) -> str:
    """Return ``stock`` with the ``_EDITS`` applied.

    Raises:
        ValueError: If a stock snippet does not occur exactly once (the upstream template changed).
    """
    tagged = stock
    for old, new in _EDITS:
        if tagged.count(old) != 1:
            raise ValueError(f"expected exactly one occurrence of {old!r} in the stock template")
        tagged = tagged.replace(old, new)
    return tagged


def main() -> None:
    """Write the tagged template and verify it against the stock one on sampled conversations."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_HERE / "templates" / "nano3_generation.jinja")
    parser.add_argument("--samples-dir", type=Path, default=_HERE / "samples")
    parser.add_argument("--check-rows", type=int, default=300, help="rows per sample file to verify")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stock = Path(hf_hub_download(MODEL, "chat_template.jinja")).read_text()
    tagged = tag_template(stock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tagged)

    tokenizer = NeMoAutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    checked = prefix_compared = 0
    for path in sorted(args.samples_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines()[: args.check_rows]:
            messages = _normalize_messages(json.loads(line)["messages"])
            text_stock = tokenizer.apply_chat_template(messages, tokenize=False, chat_template=stock)
            text_tagged = tokenizer.apply_chat_template(messages, tokenize=False, chat_template=tagged)
            if text_stock != text_tagged:
                raise AssertionError(f"tagged template renders differently for a row of {path.name}")
            out = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True, chat_template=tagged
            )
            try:
                fallback = _build_multiturn_assistant_mask(tokenizer, messages, out["input_ids"])
            except ValueError:  # the stock template rewrites history here: nothing to compare against
                fallback = None
            if fallback is not None:
                if list(out["assistant_masks"]) != fallback:
                    raise AssertionError(f"generation mask differs from the prefix mask for a row of {path.name}")
                prefix_compared += 1
            checked += 1
    logger.info(
        "wrote %s; identical renders on %d conversations, identical loss masks on the %d the stock path can mask",
        args.output,
        checked,
        prefix_compared,
    )


if __name__ == "__main__":
    main()
