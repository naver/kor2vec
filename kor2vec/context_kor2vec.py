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

import torch.nn as nn

from .kor2vec import Kor2Vec
from .trainer import SkipTrainer
from .utils.sampler.move_sampler import MovingSampler


class ContextKor2Vec(Kor2Vec):
    def __init__(self, embed_size=128, char_embed_size=64, word_seq_len=5,
                 filter_sizes=[2, 3, 4, 5], negative_sample_count=5):
        super().__init__(embed_size, char_embed_size, word_seq_len, filter_sizes, negative_sample_count)

        self.context = nn.LSTM(embed_size, embed_size, batch_first=True)

    def train(self, corpus_path=None, model_path=None, sample_path=None, sample_output_path=None,
              window_size=5, negative_sample_count=5, positive_sample_count=4,
              batch_size=1024, epochs=10, pre_sequence=False):
        if sample_path is None:
            sample_path = corpus_path + ".sampled" if sample_output_path is None else sample_output_path
            MovingSampler(corpus_path, sample_path, window=window_size, negative_size=negative_sample_count)

        # 3. Training Skip-gram with sampled words
        print("Training kor2vec")
        SkipTrainer(self, sample_path,
                    output_path=model_path, vocab=self.vocab,
                    word_seq_len=self.word_seq_len,
                    negative_sample_count=negative_sample_count,
                    positive_sample_count=positive_sample_count,
                    batch_size=batch_size, pre_sequence=pre_sequence).train(epochs)

    def forward(self, seq):
        x = self.model.forward(seq)
        x = x.unsqueeze(1) if seq.dim() < 3 else x
        x, _ = self.context.forward(x)
        x = x.squeeze(1) if seq.dim() < 3 else x
        return x
