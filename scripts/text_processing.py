"""Sections of this code are based on scikit-learn sources; scikit-learn code is covered by the following license:
New BSD License

Copyright (c) 2007–2018 The scikit-learn developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""
import scripts.utils.utils as ut
import string

from nltk import word_tokenize, PorterStemmer, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import strip_accents_ascii

from scripts import FilePaths


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def lemmatize_with_pos(self, tag):
        if tag[1].startswith('N'):
            return self.wnl.lemmatize(tag[0], wordnet.NOUN)
        elif tag[1].startswith('J'):
            return self.wnl.lemmatize(tag[0], wordnet.ADJ)
        elif tag[1].startswith('R'):
            return self.wnl.lemmatize(tag[0], wordnet.ADV)
        elif tag[1].startswith('V'):
            return self.wnl.lemmatize(tag[0], wordnet.VERB)
        else:
            return self.wnl.lemmatize(tag[0])

    def __call__(self, doc):
        text = word_tokenize(doc)
        pos_tagged_tokens = pos_tag(text)
        return [self.lemmatize_with_pos(t) for t in pos_tagged_tokens]


class StemTokenizer(object):
    def __init__(self):
        self.ps = PorterStemmer()

    def __call__(self, doc):
        return [self.ps.stem(t) for t in word_tokenize(doc)]


def lowercase_strip_accents_and_ownership(doc):
    lowercase_no_accents_doc = strip_accents_ascii(doc.lower())
    txt = lowercase_no_accents_doc.replace('"', '').replace("\'s", "").replace("\'ve", " have").replace("\'re",
                                                                                                        " are").replace(
        "\'", "").strip("`").strip()
    return txt


class WordAnalyzer(object):
    tokenizer = None
    preprocess = None
    ngram_range = None
    stemmed_stop_word_set_n = None
    stemmed_stop_word_set_uni = None
    stemmed_stop_word_set_glob = None

    @staticmethod
    def init(tokenizer, preprocess, ngram_range):
        WordAnalyzer.tokenizer = tokenizer
        WordAnalyzer.preprocess = preprocess
        WordAnalyzer.ngram_range = ngram_range

        # global stopwords
        with open(FilePaths.global_stopwords_filename, 'r') as f:
            WordAnalyzer.stemmed_stop_word_set_glob = set(WordAnalyzer.tokenizer(f.read()))

        # we want to stop some words on n>1 ngrams
        with open(FilePaths.ngram_stopwords_filename, 'r') as f:
            stop_lst = f.readlines()
            stop_lst = [x.strip() for x in stop_lst]
            n_grams_stops = []
            for stop_token in stop_lst:
                n_grams_stop = WordAnalyzer.tokenizer(stop_token)
                n_grams_stops.append(' '.join(n_grams_stop))
            WordAnalyzer.stemmed_stop_word_set_n = set(n_grams_stops)

        with open(FilePaths.unigram_stopwords_filename, 'r') as f:
            WordAnalyzer.stemmed_stop_word_set_uni = set(WordAnalyzer.tokenizer(f.read()))

    # Based on VectorizeMixin in sklearn text.py
    @staticmethod
    def analyzer(doc):
        """based on VectorizerMixin._word_ngrams in sklearn/feature_extraction/text.py,
        from scikit-learn; extended to prevent generation of n-grams containing stop words"""
        min_n, max_n = WordAnalyzer.ngram_range
        original_tokens_unstopped = WordAnalyzer.tokenizer(WordAnalyzer.preprocess(doc))
        original_tokens = [x for x in original_tokens_unstopped if x not in WordAnalyzer.stemmed_stop_word_set_glob]

        tokens = [x for x in original_tokens if x not in string.punctuation] if min_n == 1 else []

        # handle token n-grams
        if max_n > 1:
            min_phrase = max(min_n, 2)
            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_phrase, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    candidate_ngram = original_tokens[i: i + n]
                    has_punkt = False
                    for token in candidate_ngram:
                        if token in string.punctuation:
                            has_punkt = True
                            break
                    if not has_punkt:
                        tokens_append(space_join(candidate_ngram))

        return ut.stop(tokens, WordAnalyzer.stemmed_stop_word_set_uni, WordAnalyzer.stemmed_stop_word_set_n)
