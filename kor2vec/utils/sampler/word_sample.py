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

from kor2vec.model.vocab import WordVocab
import numpy
import random
import tqdm


class WordSampler:
    def __init__(self, vocab):
        word_vocab = vocab
        counter = word_vocab.freqs

        self.words, counts = list(counter.keys()), list(counter.values())

        total_token = sum(counts)
        prob = numpy.array(counts) / total_token
        positive_sample_prob = (numpy.sqrt(prob / 1e-3) + 1) * (1e-3 * prob)
        self.positive_sample_dict = {word: prob for word, prob in zip(self.words, positive_sample_prob)}

        f_w = numpy.array(counts) ** (3 / 4)
        self.negative_sample_prob = f_w / sum(f_w)

        self.negative_sample_words = []

    @staticmethod
    def random_select(p):
        return random.random() > p

    def word_select(self, words, probs):
        return [word for word, prob in zip(words, probs) if self.random_select(prob)]

    def get_negative_words(self, k):
        if len(self.negative_sample_words) > k:
            return [self.negative_sample_words.pop() for _ in range(k)]
        else:
            self.negative_sample_words = random.choices(self.words, self.negative_sample_prob, k=10000000)
            return self.get_negative_words(k)

    def get_word_samples(self, words, negative_count=1):
        word_probs = [self.positive_sample_dict[word] for word in words if word in self.positive_sample_dict]
        positive_words = self.word_select(words, word_probs)
        negative_words = [self.get_negative_words(negative_count) for _ in range(len(positive_words))]
        return zip(positive_words, negative_words)

    def get_samples(self, sentence, window_size=5, negative_count=1):
        words = sentence.split(" ")
        samples = []
        for start_index in range(len(words) - window_size + 1):
            half_window = window_size // 2
            center_word = words[start_index + half_window]
            around_words = words[start_index:start_index + half_window] + words[start_index + half_window + 1:]
            word_samples = self.get_word_samples(around_words, negative_count)
            word_samples = [(center_word, positive, "\t".join(negative)) for positive, negative in word_samples]
            samples.extend(word_samples)
        return samples

    def to_word_samples(self, corpus_path, output_path, window_size=5, negative_count=1):
        output_f = open(output_path, "w", encoding="utf-8")
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in tqdm.tqdm(f.readlines(), desc="Word sampling"):
                samples = self.get_samples(line[:-1], window_size, negative_count)
                for sample in samples:
                    output_f.write("%s\t%s\t%s\n" % sample)
        output_f.close()


if __name__ == "__main__":
    vocab_path = "../../data/movie/vocab/word.vocab"
    corpus_path = "../../data/movie/corpus.small.txt"
    output_path = "../../data/movie/word_samples.small.txt"

    word_vocab = WordVocab.load_vocab(vocab_path)
    sampler = WordSampler(word_vocab)
    sampler.to_word_samples(corpus_path, output_path, negative_count=5)
