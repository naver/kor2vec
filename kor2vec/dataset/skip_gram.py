# Copyright 2018 NAVER Corp.
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

from torch.utils.data import Dataset
from kor2vec.model.vocab import WordCharVocab

import torch
import tqdm


class SkipDataset(Dataset):
    def __init__(self, corpus_path, word_char_vocab: WordCharVocab, char_seq_len,
                 positive_sample_count, negative_sample_count, pre_sequence=False):

        self.word_char_vocab = word_char_vocab
        self.char_seq_len = char_seq_len
        self.negative_sample_count = negative_sample_count
        self.positive_sample_count = positive_sample_count
        self.pre_sequence = pre_sequence

        print("Loading Word_sample corpus")
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.texts = f.readlines()

        print("Loading corpus finished")
        self.data = []

        if self.pre_sequence:
            for text in tqdm.tqdm(self.texts):
                self.data.append(self.to_seq(text[:-1]))

    def to_seq(self, text):
        text = text.split("\t")
        center_seq = self.word_char_vocab.to_seq([text[0]], seq_len=1, word_seq_len=self.char_seq_len)[0]
        positive_seq = self.word_char_vocab.to_seq(text[1].split(" "),
                                                   seq_len=self.positive_sample_count,
                                                   word_seq_len=self.char_seq_len)
        negative_seq = self.word_char_vocab.to_seq(text[2].split(" "),
                                                   seq_len=self.negative_sample_count,
                                                   word_seq_len=self.char_seq_len)

        data = {
            "center_seq": torch.tensor(center_seq),
            "positive_seq": torch.tensor(positive_seq),
            "negative_seq": torch.tensor(negative_seq)
        }
        return data

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        if self.pre_sequence:
            return self.data[item]
        return self.to_seq(self.texts[item][:-1])
