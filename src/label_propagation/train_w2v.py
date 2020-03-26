import os
from concurrent.futures import ThreadPoolExecutor

import ndjson
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.sentiment.util import mark_negation

scores = []
reviews = []
summaries = []
with open("data/raw/Video_Games_5.json", "r") as infile:
    reader = ndjson.reader(infile)

    for review in reader:
        try:
            rating = review["overall"]
            rv = review["reviewText"]
            s = review["summary"]
        except Exception as e:
            continue
        if len(rv) > 0 and len(s) > 0:
            scores.append(rating)
            reviews.append(rv)
            summaries.append(s)


def process_review(review):
    wds = []
    sents = sent_tokenize(review)
    for sent in sents:
        ws = word_tokenize(sent)
        ws = mark_negation(ws)
        for w in ws:
            if w.isalpha() or "_" in w:
                wds.append(w.lower())
    return wds


futures = []
with ThreadPoolExecutor(max_workers=os.cpu_count()-1) as executor:
    for r in reviews:
        future = executor.submit(process_review, r)
        futures.append(future)

reviews_processed = []
for future in futures:
    res = future.result()
    reviews_processed.append(res)

model = Word2Vec(reviews_processed, min_count=1)
model.save("data/models/model.bin")
