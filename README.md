# kor2vec

OOV없이 빠르고 정확한 한국어 Embedding

## Installation
```shell
pip install git+https://github.com/naver/kor2vec.git
```
> Requirements : `tqdm`, `numpy` and support `torch >= 0.4.0`

## Introduction

한국어는 교착어라는 특성을 갖고 있습니다. 때문에 어간+어미(용언), 명사+조사 등등 다양한 형태의 수만가지의
단어 조합들을 만들어 낼 수 있는데요. 한국어를 사용하는 입장에서는 매우 편리한 특성이지만
한국어를 Embedding 해야하는 NLP 개발자들에게는 언제나 가장 큰 문제점으로 다가왔습니다.

때문에 `konlpy`나 `sentence piece`를 사용해서 한국어를 적절한 token 단위로 나눈뒤에
`Word2vec` 또는 자제적인 Embedding을 학습하여 교착어의 문제를 해결하였습니다.

하지만 이 방법에는 세가지 큰 문제점이 존재합니다.

1. 모든 inference, training 과정에 tokenizer가 붙어야 함으로 병목현상이 발생한다
2. tokenization 과정에서 의미를 잃어버리는 경우가 많다 (잘못된 tokenization)
3. 모든 단어와 문장을 cover하는 것은 불가능하다 (필연 OOV문제가 발생함)


## Solution

이러한 문제점을 해결하기 위해서 CNN을 기반으로 한 char-word 임베딩을 한국어에 적용하여
`kor2vec`을 만들게 되었습니다.

- Embedding 학습 방법 : Skip-gram based embedding training
- Char-word Encoder 모델 구조 : [Yoon Kim's Character-Aware Neural Language Modeling](https://arxiv.org/abs/1508.06615)

## Quick Start

```shell
kor2vec train -c corpus/path -o output/model.kor2vec
```

### inference
```python

from kor2vec import Kor2Vec
kor2vec = Kor2Vec.load("../model/path")

kor2vec.embedding("안녕 아이오아이야 나는 클로바에서 왔어")
>>> torch.tensor(5, 128) # embedding vector

kor2vec.embedding("나는 도라에몽이라고 해 반가워", numpy=True)
>>> numpy.array(4, 128) # numpy embedding vector

input = kor2vec.to_seqs(["안녕 나는 뽀로로라고 해", "만나서 반가워 뽀로로"], seq_len=4)
kor2vec.forward(input)
>> torch.tensor([2, 4, 128])
```

### training

```python
from kor2vec import Kor2vec

kor2vec = Kor2Vec(embed_size=128)

kor2vec.train("../path/corpus", batch_size=128) # takes some time

kor2vec.save("../mode/path") # saving embedding
```

### with pytorch

```python

import torch.nn as nn
from kor2vec import Kor2vec

kor2vec = Kor2Vec.load("../model/path")
# or kor2vec = SejongVector()

lstm = nn.LSTM(128, 64, batch_first=True)
dense = nn.Linear(64, 1)

# Make tensor input
sentences = ["이 영화는 정말 대박이에요", "우와 진짜 재미있었어요"]

x = kor2vec.to_seqs(sentences, seq_len=10)
# >>> tensor(batch_size, seq_len, char_seq_len)

x = kor2vec(x) # tensor(batch_size, seq_len, 128)
_, (x, xc) = lstm(x) # tensor(batch_size, 64)
x = dense(x) # tensor(batch_size, 1)

```

## License

```
Copyright 2018 NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Author
김준성 (Naver Clova AI intern) : codertimo@gmail.com