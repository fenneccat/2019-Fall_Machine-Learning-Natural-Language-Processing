# Machine Learning & Natural Language Processing course in KAIST 2019 FALL

## Word2Vec
Implement Word2Vec - Skip-gram with negative sampling.
* Skip gram class is implemented from base
* Understand how Skip-gram works

## BoW Classification
Implement BOW vectorizer
* BOW vectorizer is implemented from base (without `scikit-learn`)
* Apply multiple classifier (Logistic regression, MultinomialNB, MLPClassifier) and ensemble classifiers to get better classification result

## Naive Bayes
Implement Naive Bayes model
* Calculate prior, likelihood
* Add one smoothing

## N-gram Language Model
Implement N-gram language model
* precalculate probability distribution for vocabulary of the training corpus (use `NLTK.ngrams`
* Predict most likely next token with given previous tokens
* Measure perplexity to test LM performance

## Recurrent Neural Network
Implement Recurrent Neural Network model
* Build RNN model with tensorflow

## Wordpiece RNN (Korean Data)
Implement sub-word RNN model to avoid overfitting problem
* Breaking word into sub-word => Byte-pair encoding
* Compare performance of RNN LM with three different setting
	- word token
	- Byte-pair encoding
	- SentencePiece (Google)
