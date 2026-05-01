"""Shared NLP model state — loaded once at startup, accessed by pipeline modules."""

nlp_model = None
sentence_model = None
word2vec_model = None       # gensim Word2Vec trained on combined job + resume corpus
tfidf_vectorizer = None     # TfidfVectorizer fitted on job corpus for meaningful IDF weights
