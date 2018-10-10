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

from .model.vocab import WordCharVocab
from .model import WordCNNEmbedding

from .trainer import SkipTrainer

from .utils.build_vocab import get_all_char
from .utils.sampler.move_sampler import MovingSampler

import torch
import torch.nn as nn


class Kor2Vec(nn.Module):
    """
    Author by junseong.kim, Clova AI Intern
    Written at 2018.09.01

    Any questions, suggestion and feedback are welcomed anytime :)
    codertimo@gmail.com / junseong.kim@navercorp.com
    """

    def __init__(self, embed_size=128, char_embed_size=64, word_seq_len=5,
                 filter_sizes=[2, 3, 4, 5], negative_sample_count=5):
        super().__init__()

        self.word_seq_len = word_seq_len
        self.embed_size = embed_size
        self.negative_sample_count = negative_sample_count
        self.char_embed_size = char_embed_size
        self.filter_sizes = filter_sizes
        self.vocab = WordCharVocab(get_all_char())

        self.model = WordCNNEmbedding(embed_size=self.embed_size,
                                      char_embed_size=self.char_embed_size,
                                      vocab_size=len(self.vocab),
                                      char_seq_len=self.word_seq_len,
                                      filter_sizes=self.filter_sizes)

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

    def embedding(self, sentence, seq_len=None, numpy=False, with_len=False):
        if isinstance(sentence, str):
            x = self.to_seq(sentence, seq_len, with_len=with_len)
        elif isinstance(sentence, list):
            x = self.to_seqs(sentence, seq_len, with_len=with_len)
        else:
            x = None

        if with_len:
            x, x_seq_len = x

        x = self.forward(x)
        x = x if not numpy else x.detach().numpy()
        return (x, x_seq_len) if with_len else x

    def to_seq(self, sentence, seq_len=None, numpy=False, with_len=False):
        x, x_seq_len = self.vocab.to_seq(sentence, seq_len=seq_len, word_seq_len=self.word_seq_len, with_len=True)
        x = torch.tensor(x).to(self.get_device())
        x = x if not numpy else x.numpy()
        return (x, x_seq_len) if with_len else x

    def to_seqs(self, sentences, seq_len, numpy=False, with_len=False):
        sequences = [self.to_seq(sentence, seq_len, with_len=True) for sentence in sentences]
        seqs = torch.stack([seq for seq, _seq_len in sequences], dim=0)
        seqs = seqs if not numpy else seqs.numpy()
        seq_lens = [_seq_len for seq, _seq_len in sequences]
        return (seqs, seq_lens) if with_len else seqs

    def forward(self, seq):
        return self.model.forward(seq)

    def get_device(self):
        return next(self.parameters()).device

    @staticmethod
    def load(path):
        return torch.load(path, map_location={'cuda:0': 'cpu'})

    def save(self, path):
        origin_device = self.get_device()
        torch.save(self.to(torch.device("cpu")), path)
        self.to(origin_device)
