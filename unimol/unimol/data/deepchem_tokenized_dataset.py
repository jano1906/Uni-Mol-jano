# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from unicore.data import BaseWrapperDataset
from unimol.data.deep_chem_tokenizer.deepchem_tokenizer import SmilesTokenizer
import torch

class DeepchemTokenizedDataset(BaseWrapperDataset):
    def __init__(self, dataset, max_seq_len):
        self.dataset = dataset
        self.tokenizer = SmilesTokenizer()
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        ret = self.tokenizer.encode(self.dataset[idx])
        assert len(ret) <= self.max_seq_len, f"{len(ret)} is larger than {self.max_seq_len}"
        return torch.tensor(ret)