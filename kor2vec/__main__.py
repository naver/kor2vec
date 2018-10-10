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

from kor2vec import Kor2Vec
from kor2vec import ContextKor2Vec
import argparse

main_parser = argparse.ArgumentParser()
subparsers = main_parser.add_subparsers()

parser = subparsers.add_parser("train", help="train mode")
parser.add_argument('-c', '--corpus_path', type=str, required=True)
parser.add_argument('-o', '--output_path', type=str, default="model.kor2vec")
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('-e', '--epochs', default=10, type=int)
parser.add_argument('--pre_seq', default=False, type=bool)

# Word sampler arguments
parser.add_argument('--sampled', default=False, type=bool)
parser.add_argument('--window_size', default=5, type=int)
parser.add_argument('--sample_count', default=5, type=int)
parser.add_argument('--sample_output_path', default=None, type=str)

args = main_parser.parse_args()


def kor2vec_main():
    # Define Model and bind to NSML
    model = Kor2Vec()

    # training model
    model.train(
        corpus_path=args.corpus_path if not args.sampled else None,
        model_path=args.output_path,
        sample_path=args.corpus_path if args.sampled else None,
        window_size=args.window_size,
        negative_sample_count=args.window_size - 1,
        positive_sample_count=args.window_size - 1,
        sample_output_path=args.sample_output_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        pre_sequence=args.pre_seq)


def kor2vec_context_main():
    # Define Model and bind to NSML
    model = ContextKor2Vec()

    # training model
    model.train(
        corpus_path=args.corpus_path if not args.sampled else None,
        model_path=args.output_path,
        sample_path=args.corpus_path if args.sampled else None,
        window_size=args.window_size,
        negative_sample_count=args.window_size - 1,
        positive_sample_count=args.window_size - 1,
        sample_output_path=args.sample_output_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        pre_sequence=args.pre_seq)
