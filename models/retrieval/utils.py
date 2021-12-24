# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Pre-processing
Post-processing utilities for question answering.
"""
import logging
import os
import pickle
import random

import numpy as np
import torch
from konlpy.tag import Mecab
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import is_torch_available

logger = logging.getLogger(__name__)

mecab = Mecab()


def tokenize(text):
    # return text.split(" ")
    return mecab.morphs(text)


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BM25(object):
    def __init__(self, tokenizer=tokenize, ngram_range=(1, 2), max_features=50000, b=0.75, k1=1.6):
        # 만약 tfidfvectorizer가 있으면 불러와서 저장, fit함수 저장
        tfidfv_path = '/opt/ml/input/data/tfidv.bin'
        if os.path.isfile(tfidfv_path):
            with open(tfidfv_path, "rb") as file:
                self.vectorizer = pickle.load(file)
            self.is_fit = True
            print('load the tfidfv')
        else:
            self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False,
                                              tokenizer=tokenize, ngram_range=(1, 2), max_features=max_features)
            self.is_fit = False
        self.b = b
        self.k1 = k1

    def fit_transform(self, context):
        if not self.is_fit:
            self.vectorizer.fit(context)
            self.is_fit = True
        y = super(TfidfVectorizer, self.vectorizer).transform(context)
        self.avdl = y.sum(1).mean()

        b, k1, avdl = self.b, self.k1, self.avdl
        len_y = y.sum(1).A1

        y = y.tocsc()
        denom = y + (k1 * (1 - b + b * len_y / avdl))[:, None]
        numer = y * (k1 + 1)
        p_embedding = (numer / denom)
        return csr_matrix(p_embedding, dtype=np.float16)

    def fit(self, X):
        """ Fit IDF to documents X """
        if not self.is_fit:
            self.vectorizer.fit(X)
            self.is_fit = True
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, X):
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        assert sparse.isspmatrix_csr(X)
        idf = self.vectorizer._tfidf.transform(X, copy=False)
        # idf.todense()
        # idf.data -= 1
        # idf.tocsc()
        return idf