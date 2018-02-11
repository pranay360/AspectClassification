# Aspect Classification

## Required Libraries for python:
NLTK, Sklearn, Pandas, Gensim, Numpy, matplotlib


## Other Requirements
Word2Vec File : GoogleNews-vectors-negative300.bin 3.6 GBs uncompressed
Link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit 

## Data Requirements:
aspect_annoated_file.txt
aspect_words_1k_restaurant_reviews
for any other domain please ensure same format as aspect_annoated_file.txt

## About the Problem:
Aspect Classification can be used in scenarios where for example a classification like sentiment analysis is being done, the aspects in the text can be used to find the actionable items that makes it belong to a class. For example if a review has a negative sentiment, then what were the concerned actionable items that made the review negative. In this way customer service oriented companies can actually work on some "aspects" which makes a review positive or negative or belong to a particular class.

## Approach:
Sentences, aspects were extracted from the annotated file. Each sentence was POS Tagged. Then features for each word in a sentence were extracted, considering the context through previous and next words. Currently only 2 previous and next words along with their POS are take n as context. Finally for every word in each sentence a context feature vector is generated , which can be trained against the corresponding Aspect. Note training and prediction is done at a word by word level. I also tried to concatenate word embeddings of a word to it’s above mentioned feature vector, but there was not significant improvement in the results, therefore dropped them as additional features. Also, the pretrained embedding used doesn’t contain many stopwords and special words, for such words a random embedding was generated, which obviously was a wrong move. Training our own embeddings might help.

## Features and Algorithms
The main focus throughout was what features can be used to tell whether a word is an aspect or not. Firstly the word , its letter level features -first word capital? Upper case? Lower case? and its POS tag and Finally for the context the “n” previous and next  words  were taken (currently n can be  1 or 2 ) and their corresponding POS tags are taken as features,  and Aspect of previous word is taken because mostly aspects like dishes and food items are listed together by the reviewer so, if previous word is an aspect then the next word tends to be an aspect. POS tag of the word and previous and next words were crucial indicators of being an aspect. These all features are converted into an one hot encoded fixed length vector. At first machine learning algorithms like Naive Bayes, Logistic Regression but they gave low ROC Area under the curve score due to high false positive rate, hence to handle the sparsity and imbalance Support vector classifier was used as its good linear classifier for binary classification with class imbalance.
