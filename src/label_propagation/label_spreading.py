import warnings
warnings.filterwarnings("ignore")

import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import opinion_lexicon
from sklearn.semi_supervised import LabelSpreading

model = Word2Vec.load("data/models/model.bin")
vocab = list(model.wv.vocab.keys())

positive_wds = set(opinion_lexicon.positive())
negative_wds = set(opinion_lexicon.negative())

positive_wds_with_negation = positive_wds.union({wd + "_neg" for wd in
                                                 negative_wds})
negative_wds_with_negation = negative_wds.union({wd + "_neg" for wd in
                                                 positive_wds})


def label_word(wd):
    if wd in positive_wds_with_negation:
        return 1
    elif wd in negative_wds_with_negation:
        return 0
    else:
        return -1

X = np.array([model[wd] for wd in vocab])
labels = [label_word(wd) for wd in vocab]
label_spread = LabelSpreading(kernel='knn', alpha=0.8, n_jobs=3)
label_spread.fit(X, labels)

wds = [wd for wd in vocab if wd not in positive_wds_with_negation and wd not
       in negative_wds_with_negation]


def get_polarity_proba(w):
    res = label_spread.predict_proba([model[w]])[0]
    pos, neg = res[0], res[1]
    if pos > neg:
        return pos
    else:
        return neg * -1


probs = (get_polarity_proba(w) for w in wds)


with open("data/processed/words_with_sentiment.tsv", "w") as outfile:
    lines = []
    for e in zip(wds, probs):
        wd = e[0]
        proba = e[1]
        if proba > 1:
            o = wd + "\tpositive\t" + str(proba)
        else:
            o = wd + "\tnegative\t" + str(proba * -1)
        lines.append(o)
    lines = "\n".join(lines)
    outfile.write(lines)
