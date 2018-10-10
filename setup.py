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

from setuptools import setup, find_packages

setup(
    name="kor2vec",
    version="1.0.1",
    description="Quality Korean Word Embedding without UNK",
    author='Junseong Kim',
    author_email='codertimo@gmail.com',
    packages=find_packages(),
    install_requires=[
        "torch>=0.4.0",
        "tqdm",
        "numpy"
    ],
    entry_points={
        'console_scripts': [
            'kor2vec = kor2vec.__main__:kor2vec_main',
            'kor2vec-context = kor2vec.__main__:kor2vec_context_main'
        ]
    },
    # package_data={'kor2vec': ['pretrained/model/sejong.kor2vec']}
)
