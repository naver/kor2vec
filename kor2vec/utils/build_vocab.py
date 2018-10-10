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

from kor2vec.model.vocab import WordVocab, WordCharVocab


def building_vocab(corpus_path, save_vocab_dir=None):
    with open(corpus_path, "r", encoding="utf-8") as f:
        texts = f.readlines()

    vocab = WordVocab(texts)
    print("WORD VOCAB SIZE:", len(vocab))

    if save_vocab_dir is not None:
        word_vocab_path = save_vocab_dir + "word.vocab"
        vocab.save_vocab(word_vocab_path)
        print("WORD VOCAB SAVED:", word_vocab_path)

    return vocab


def get_all_char():
    koreans = [chr(i) for i in range(44032, 55203)]
    korean_chars = [chr(i) for i in range(ord("ㄱ"), ord("ㅎ") + 22)]
    special_chars = [chr(i) for i in range(ord("!"), ord("~"))]
    all_chars = koreans + korean_chars + special_chars
    return all_chars
