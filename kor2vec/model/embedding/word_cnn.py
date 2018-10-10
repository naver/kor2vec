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

import torch
import torch.nn as nn
import pickle


class WordCNNEmbedding(nn.Module):
    """
    Convolution based Char-Word Embedding
    developed by junseong.kim 2018.08 for Naver Clova duplex project

    Reference : Character-Aware Neural Language Models,
    Yoon Kim 2015 (https://arxiv.org/abs/1508.06615)

    Module Init:
        embed_size: Char Embedding Size
        hidden_size: hidden_size of convolution multiplication
        char_vocab_dict: dictionary of {"char": index } ex {"a": 31, "b": 32}
        char_seq_len : char_length of each word
        filter_sizes: conv filter sizes

    input:
        - tensor(batch_size, seq_len, char_seq_len)
        - tensor(batch_size, char_seq_len)

    output:
        - tensor(batch_size, seq_len, self.output_size)
        - tensor(batch_size, self.output_size)

    """

    def __init__(self, embed_size, char_embed_size, vocab_size, char_seq_len, filter_sizes=[2, 3, 4, 5]):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = char_seq_len
        self.embed_size = embed_size
        self.char_embed_size = char_embed_size

        if embed_size % len(filter_sizes) != 0:
            raise Exception("embed_size is not dividable with filter_size: embed_size:%d, filter_size:%d" % (
                embed_size, len(filter_sizes)))
        self.hidden_size = embed_size // len(filter_sizes)

        # Character Embedding
        self.embedding = nn.Embedding(self.vocab_size, self.char_embed_size, padding_idx=0)

        # Multiple-width Convolution Filters
        self.convs = nn.ModuleList([nn.Conv1d(self.char_embed_size, self.hidden_size, width) for width in filter_sizes])
        self.pools = nn.ModuleList([nn.MaxPool1d(self.seq_len - width + 1) for width in filter_sizes])

    def forward(self, x):
        encoder_embed = self.embedding(x)

        # 4-dim (batch_size, word_seq_len, char_seq, embed_dim)
        if encoder_embed.dim() == 4:
            batch_size, seq_len, char_seq_len, embed_size = [encoder_embed.size(i) for i in range(4)]
            words = encoder_embed.view(batch_size * seq_len, char_seq_len, embed_size)
            embed_words = self.word_embedding(words)
            output = embed_words.view(batch_size, seq_len, self.embed_size)
            return output

        # 3-dim (batch_size, char_seq, embed_dim)
        return self.word_embedding(encoder_embed)

    def word_embedding(self, word):
        conv_outputs = []
        for conv, pool in zip(self.convs, self.pools):
            x = conv(word.transpose(1, -1))
            x = pool(x)
            conv_outputs.append(x.squeeze(-1))
        return torch.cat(conv_outputs, dim=-1)

    @staticmethod
    def load_model(path):
        with open(path, "rb") as f:
            return pickle.load(path)

    def save_model(self, path):
        with open(path, "wb") as f:
            return pickle.dump(self, f)
