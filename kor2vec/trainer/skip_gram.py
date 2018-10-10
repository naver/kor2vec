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

from torch.optim import Adam
import tqdm

from ..dataset import SkipDataset
from ..model import SkipGram

from torch.utils.data import DataLoader

import torch
import torch.nn as nn


class SkipTrainer:
    def __init__(self, kor2vec_model, corpus_path, output_path, vocab, batch_size, pre_sequence,
                 word_seq_len, negative_sample_count, positive_sample_count):
        self.kor2vec = kor2vec_model
        self.model = SkipGram(embedding=self.kor2vec)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = Adam(self.model.parameters())

        dataset = SkipDataset(corpus_path, vocab,
                              char_seq_len=word_seq_len,
                              negative_sample_count=negative_sample_count,
                              pre_sequence=pre_sequence,
                              positive_sample_count=positive_sample_count)
        self.data_loader = DataLoader(dataset, batch_size=batch_size)
        self.output_path = output_path

    def train(self, epochs):
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        print("CUDA Available/count:", torch.cuda.is_available(), torch.cuda.device_count())
        print("training on ", self.device)

        dp_model = nn.DataParallel(self.model)
        dp_model.to(self.device)

        for epoch in range(epochs):
            avg_loss = self.forward(epoch)
            torch.save(self.kor2vec, self.output_path + ".ep%d" % epoch)

    def forward(self, epoch, train=True, verbose=True, log_code="train"):
        avg_loss, total_correct, total_nelement = 0.0, 0.0, 0

        iterator = tqdm.tqdm(enumerate(self.data_loader), total=len(self.data_loader), desc="EP %d" % epoch)
        for step, data in iterator:
            data = {key: value.to(self.device) for key, value in data.items()}

            loss = self.model.forward(data["center_seq"], data["positive_seq"], data["negative_seq"])
            avg_loss += loss.item()

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            output_log = {
                "epoch": epoch,
                "step": step,
                "%s_loss" % log_code: loss.item(),
                "device": str(data["center_seq"].device)
            }

        avg_loss /= len(self.data_loader)
        output_log = {
            "epoch": epoch,
            "%s_ep_loss" % log_code: avg_loss,
        }

        if verbose:
            print(output_log)

        output_log["step"] = epoch
        return avg_loss
