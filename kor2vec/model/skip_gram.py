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
import torch.nn.functional as fnn


class SkipGram(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, u_pos, v_pos, v_neg):
        batch_size = u_pos.size(0)
        positive_size = v_pos.size(1)
        negative_size = v_neg.size(1)

        embed_u = self.embedding(u_pos)
        embed_v = self.embedding(v_pos)

        score = torch.bmm(embed_v, embed_u.unsqueeze(2)).squeeze(-1)
        score = torch.sum(score, dim=1) / positive_size
        log_target = fnn.logsigmoid(score).squeeze()

        neg_embed_v = self.embedding(v_neg)

        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze(-1)
        neg_score = torch.sum(neg_score, dim=1) / negative_size
        sum_log_sampled = fnn.logsigmoid(-1 * neg_score).squeeze()

        loss = log_target + sum_log_sampled

        return -1 * loss.sum() / batch_size
