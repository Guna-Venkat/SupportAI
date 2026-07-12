"""
collator.py
===========
Dynamic padding data collator for transformer batching.
"""

from typing import Any

import torch


class DynamicPaddingCollator:
    """Collates batch sequences by dynamically padding them to the batch maximum length."""

    def __init__(self, pad_token_id: int = 0, pad_to_multiple_of: int | None = None) -> None:
        """Initialises the collator.

        Args:
            pad_token_id: The ID of the padding token used by the tokenizer.
            pad_to_multiple_of: If set, will pad the sequence to a multiple of the provided value
                (e.g., 8 for faster mixed-precision tensor core operations).
        """
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Pads and stacks features into batch tensors.

        Args:
            features: List of dictionaries containing "input_ids", "attention_mask",
                and optional "label".

        Returns:
            Dictionary containing "input_ids", "attention_mask", and "labels" tensors.
        """
        # 1. Retrieve raw sequences and labels
        input_ids_list = [feature["input_ids"] for feature in features]
        attention_mask_list = [feature["attention_mask"] for feature in features]

        # 2. Compute maximum sequence length in the current batch
        max_len = max(len(ids) for ids in input_ids_list)

        if self.pad_to_multiple_of is not None:
            # Round up max_len to the nearest multiple of pad_to_multiple_of
            remainder = max_len % self.pad_to_multiple_of
            if remainder > 0:
                max_len += self.pad_to_multiple_of - remainder

        # 3. Perform dynamic padding
        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(input_ids_list, attention_mask_list, strict=True):
            pad_len = max_len - len(ids)

            # Pad input_ids with pad_token_id
            padded_ids = torch.cat(
                [ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)]
            )
            padded_input_ids.append(padded_ids)

            # Pad attention_mask with 0
            padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            padded_attention_masks.append(padded_mask)

        # 4. Stack into batched tensors
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
        }

        # 5. Handle optional labels
        if "label" in features[0]:
            labels = [feature["label"] for feature in features]
            batch["labels"] = torch.stack(labels)

        return batch
