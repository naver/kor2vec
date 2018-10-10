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

import unittest
from kor2vec import Kor2Vec, ContextKor2Vec


class Kor2VecTestCase(unittest.TestCase):
    def setUp(self):
        self.kor2vec = Kor2Vec()

    def test_to_seq(self):
        t = self.kor2vec.to_seq("안녕 아이오아이야 나는 클로바에서 왔어")
        self.assertEqual(str(t.size()), "torch.Size([5, 5])")

    def test_to_seq_numpy(self):
        t = self.kor2vec.to_seq("안녕 아이오아이야 나는 클로바에서 왔어", numpy=True)
        self.assertEqual(str(t.shape), "(5, 5)")

    def test_embedding(self):
        t = self.kor2vec.embedding("안녕 아이오아이야 나는 클로바에서 왔어")
        self.assertEqual(str(t.size()), "torch.Size([5, 128])")

    def test_embedding_numpy(self):
        t = self.kor2vec.embedding("나는 도라에몽이라고 해 반가워", numpy=True)
        self.assertEqual(str(t.shape), "(4, 128)")

    def test_to_seqs(self):
        t = self.kor2vec.to_seqs(["안녕 나는 뽀로로라고 해", "만나서 반가워 뽀로로"], seq_len=4)
        self.assertEqual(str(t.size()), "torch.Size([2, 4, 5])")

    def test_to_seqs_numpy(self):
        t = self.kor2vec.to_seqs(["안녕 나는 뽀로로라고 해", "만나서 반가워 뽀로로"], seq_len=4, numpy=True)
        self.assertEqual(str(t.shape), "(2, 4, 5)")

    def test_forward(self):
        t = self.kor2vec.to_seq("안녕 아이오아이야 나는 클로바에서 왔어")
        t = self.kor2vec.forward(t)
        self.assertEqual(str(t.size()), "torch.Size([5, 128])")

    def test_forward_seqs(self):
        t = self.kor2vec.to_seqs(["안녕 나는 뽀로로라고 해", "만나서 반가워 뽀로로"], seq_len=4)
        t = self.kor2vec.forward(t)
        self.assertEqual(str(t.size()), "torch.Size([2, 4, 128])")

    def test_to_seq_with_len(self):
        t, seq_len = self.kor2vec.to_seq("안녕 아이오아이야 나는 클로바에서 왔어", with_len=True)
        self.assertEqual(str(t.size()), "torch.Size([5, 5])")
        self.assertEqual(seq_len, 5)

    def test_to_seq_numpy_with_len(self):
        t, seq_len = self.kor2vec.to_seq("안녕 아이오아이야 나는 클로바에서 왔어", numpy=True, with_len=True)
        self.assertEqual(str(t.shape), "(5, 5)")
        self.assertEqual(seq_len, 5)

    def test_embedding_with_len(self):
        t, seq_len = self.kor2vec.embedding("안녕 아이오아이야 나는 클로바에서 왔어", with_len=True)
        self.assertEqual(str(t.size()), "torch.Size([5, 128])")
        self.assertEqual(seq_len, 5)

    def test_embedding_numpy_with_len(self):
        t, seq_len = self.kor2vec.embedding("나는 도라에몽이라고 해 반가워", numpy=True, with_len=True)
        self.assertEqual(str(t.shape), "(4, 128)")
        self.assertEqual(seq_len, 4)

    def test_to_seqs_with_len(self):
        t, seq_len = self.kor2vec.to_seqs(["안녕 나는 뽀로로라고 해", "만나서 반가워 뽀로로"], seq_len=4, with_len=True)
        self.assertEqual(str(t.size()), "torch.Size([2, 4, 5])")
        self.assertEqual(seq_len, [4, 3])

    def test_to_seqs_numpy_with_len(self):
        t, seq_len = self.kor2vec.to_seqs(["안녕 나는 뽀로로라고 해", "만나서 반가워 뽀로로"], seq_len=4, numpy=True, with_len=True)
        self.assertEqual(str(t.shape), "(2, 4, 5)")
        self.assertEqual(seq_len, [4, 3])

    def test_forward_with_len(self):
        t, seq_len = self.kor2vec.to_seq("안녕 아이오아이야 나는 클로바에서 왔어", with_len=True)
        t = self.kor2vec.forward(t)
        self.assertEqual(str(t.size()), "torch.Size([5, 128])")
        self.assertEqual(seq_len, 5)

    def test_forward_seqs_with_len(self):
        t, seq_len = self.kor2vec.to_seqs(["안녕 나는 뽀로로라고 해", "만나서 반가워 뽀로로"], seq_len=4, with_len=True)
        t = self.kor2vec.forward(t)
        self.assertEqual(str(t.size()), "torch.Size([2, 4, 128])")
        self.assertEqual(seq_len, [4, 3])


class ContextKor2VecTestCase(unittest.TestCase):
    def setUp(self):
        self.kor2vec = ContextKor2Vec()

    def test_to_seq(self):
        t = self.kor2vec.to_seq("안녕 아이오아이야 나는 클로바에서 왔어")
        self.assertEqual(str(t.size()), "torch.Size([5, 5])")

    def test_to_seq_numpy(self):
        t = self.kor2vec.to_seq("안녕 아이오아이야 나는 클로바에서 왔어", numpy=True)
        self.assertEqual(str(t.shape), "(5, 5)")

    def test_embedding(self):
        t = self.kor2vec.embedding("안녕 아이오아이야 나는 클로바에서 왔어")
        self.assertEqual(str(t.size()), "torch.Size([5, 128])")

    def test_embedding_numpy(self):
        t = self.kor2vec.embedding("나는 도라에몽이라고 해 반가워", numpy=True)
        self.assertEqual(str(t.shape), "(4, 128)")

    def test_to_seqs(self):
        t = self.kor2vec.to_seqs(["안녕 나는 뽀로로라고 해", "만나서 반가워 뽀로로"], seq_len=4)
        self.assertEqual(str(t.size()), "torch.Size([2, 4, 5])")

    def test_to_seqs_numpy(self):
        t = self.kor2vec.to_seqs(["안녕 나는 뽀로로라고 해", "만나서 반가워 뽀로로"], seq_len=4, numpy=True)
        self.assertEqual(str(t.shape), "(2, 4, 5)")

    def test_forward(self):
        t = self.kor2vec.to_seq("안녕 아이오아이야 나는 클로바에서 왔어")
        t = self.kor2vec.forward(t)
        self.assertEqual(str(t.size()), "torch.Size([5, 128])")

    def test_forward_seqs(self):
        t = self.kor2vec.to_seqs(["안녕 나는 뽀로로라고 해", "만나서 반가워 뽀로로"], seq_len=4)
        t = self.kor2vec.forward(t)
        self.assertEqual(str(t.size()), "torch.Size([2, 4, 128])")

    def test_to_seq_with_len(self):
        t, seq_len = self.kor2vec.to_seq("안녕 아이오아이야 나는 클로바에서 왔어", with_len=True)
        self.assertEqual(str(t.size()), "torch.Size([5, 5])")
        self.assertEqual(seq_len, 5)

    def test_to_seq_numpy_with_len(self):
        t, seq_len = self.kor2vec.to_seq("안녕 아이오아이야 나는 클로바에서 왔어", numpy=True, with_len=True)
        self.assertEqual(str(t.shape), "(5, 5)")
        self.assertEqual(seq_len, 5)

    def test_embedding_with_len(self):
        t, seq_len = self.kor2vec.embedding("안녕 아이오아이야 나는 클로바에서 왔어", with_len=True)
        self.assertEqual(str(t.size()), "torch.Size([5, 128])")
        self.assertEqual(seq_len, 5)

    def test_embedding_numpy_with_len(self):
        t, seq_len = self.kor2vec.embedding("나는 도라에몽이라고 해 반가워", numpy=True, with_len=True)
        self.assertEqual(str(t.shape), "(4, 128)")
        self.assertEqual(seq_len, 4)

    def test_to_seqs_with_len(self):
        t, seq_len = self.kor2vec.to_seqs(["안녕 나는 뽀로로라고 해", "만나서 반가워 뽀로로"], seq_len=4, with_len=True)
        self.assertEqual(str(t.size()), "torch.Size([2, 4, 5])")
        self.assertEqual(seq_len, [4, 3])

    def test_to_seqs_numpy_with_len(self):
        t, seq_len = self.kor2vec.to_seqs(["안녕 나는 뽀로로라고 해", "만나서 반가워 뽀로로"], seq_len=4, numpy=True, with_len=True)
        self.assertEqual(str(t.shape), "(2, 4, 5)")
        self.assertEqual(seq_len, [4, 3])

    def test_forward_with_len(self):
        t, seq_len = self.kor2vec.to_seq("안녕 아이오아이야 나는 클로바에서 왔어", with_len=True)
        t = self.kor2vec.forward(t)
        self.assertEqual(str(t.size()), "torch.Size([5, 128])")
        self.assertEqual(seq_len, 5)

    def test_forward_seqs_with_len(self):
        t, seq_len = self.kor2vec.to_seqs(["안녕 나는 뽀로로라고 해", "만나서 반가워 뽀로로"], seq_len=4, with_len=True)
        t = self.kor2vec.forward(t)
        self.assertEqual(str(t.size()), "torch.Size([2, 4, 128])")
        self.assertEqual(seq_len, [4, 3])
