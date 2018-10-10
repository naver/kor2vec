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

import tqdm
import random


class MovingSampler:
    def __init__(self, corpus_path, output_path, window=5, negative_size=5, negative_skip_lines=100):
        self.window = window
        self.negative_size = negative_size
        self.mid_index = window // 2
        self.negative_skip_lines = negative_skip_lines

        output_file = open(output_path, "w", encoding="utf-8")
        with open(corpus_path, 'r', encoding="utf-8") as f:
            print("Reading Corpus lines")
            lines = f.readlines()
            lines = [line.split() for line in tqdm.tqdm(lines, desc="Spliting Lines")]
            for i, line in tqdm.tqdm(enumerate(lines), total=len(lines), desc="Corpus Sampling"):
                samples = self.line_separate(i, lines)
                for center, positive, negative in samples:
                    output_file.write("%s\t%s\t%s\n" % (center, " ".join(positive), " ".join(negative)))

        output_file.close()

    def negative_words(self, lines, k=100):
        words = []
        count = 0
        while len(words) <= k:
            negative_line = lines[random.randint(0, len(lines) - 1)]
            words.extend([word for word in negative_line if word != ""])
            count += 1

        return words

    def line_separate(self, step, lines):
        samples = []
        line = lines[step]
        negative_words = self.negative_words(lines, k=(len(line) - self.window) * self.negative_size)

        for i in range(len(line) - self.window):
            center = line[i + self.mid_index]
            positive = line[i:i + self.mid_index] + line[i + self.mid_index + 1: i + self.window]
            negative = negative_words[i * self.negative_size:(i + 1) * self.negative_size]
            samples.append((center, positive, negative))
        return samples


if __name__ == "__main__":
    clean_corpus_path = "corpus/clean/news2017.corpus.%s.txt"
    output_corpus_path = "corpus/sample/news2017.corpus.%s.txt"
    sampler = MovingSampler(clean_corpus_path, output_corpus_path)
