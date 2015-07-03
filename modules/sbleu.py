from __future__ import division
import numpy as np

def my_log(n):
    if n == 0:
        return 0
    return np.log(n)

def get_ngrams(sentence, n):

    words = sentence.strip().split( )

    ngrams = []

    for i in xrange(0, len(words)-n+1):
        ngram = ""
        for m in xrange(0,n):
            ngram += words[i+m]
        ngrams.append(ngram);

    return ngrams

def bleu(hypothesis, reference, bleu_n):

    brevity_penalty = 1.0;
    bleu_score = 0;

    bleu = [];

    for n in xrange(0,bleu_n):

        hypothesis_ngrams = get_ngrams(hypothesis, n+1);
        reference_ngrams = get_ngrams(reference, n+1);

        count = 0
        for ngram in hypothesis_ngrams:
            if ngram in reference_ngrams:
                count += 1

        total = len(hypothesis_ngrams)

        if count != 0:
            precision = count/total
        else:
            precision = 0

        bleu.append(precision)

    brevity_penalty = min(1, np.exp(1 - len(reference)/len(hypothesis)))

    sum = 0
    for i in xrange(0,bleu_n):
        sum += my_log(bleu[i])

    bleu_score = brevity_penalty * np.exp(sum)

    if 0 in bleu[:bleu_n]:
        bleu_score = 0        
    
    return bleu_score*100