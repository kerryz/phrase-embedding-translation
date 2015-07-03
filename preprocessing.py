from gensim import corpora, models, similarities
import numpy as np
import gensim, bz2, sys, os
import modules.sbleu as sbleu
from modules.lazy_file_reader import LazyFileReader

"""

preprocessing.py

Program that generates all data needed for main.py

@corpus_reference_file_name - path to source language corpus file
@corpus_target_file_name - path to target language corpus file
@n_best_list_file_name - path to n best translation of the source sentences
@reference_file_name - path to reference translation of the source sentences

"""

class MyCorpus(object):

    def __init__(self, corpus_reference, corpus_target):
        self.corpus_reference = corpus_reference
        self.corpus_target = corpus_target

    def __iter__(self):
        for corpus_reference_line, corpus_target_line in zip(self.corpus_reference, self.corpus_target):
            yield corpus_reference_line + " " + corpus_target_line

class MyBowCorpus(object):

    def __init__(self, dictionary, corpus):
        self.corpus = corpus
        self.dictionary = dictionary

    def __iter__(self):
        for line in self.corpus:
            yield self.dictionary.doc2bow(line.lower().split())

def build_corpus_dictionary(corpus):

    dictionary = corpora.Dictionary(line.lower().split() for line in corpus)

    dictionary.compactify()
    dictionary.save('data/dictionary.dict') # store the dictionary, for future reference

    corpus = MyBowCorpus(dictionary, corpus)
    corpora.MmCorpus.serialize('data/corpus.mm', corpus) # store the corpus for future reference

    return corpus, dictionary

def run_lda(corpus, num_topics):

    lda = models.LdaMulticore(corpus, num_topics)
    lda.save('data/model.lda')

    return lda

def save_topics(lda, dictionary):

    f = open("data/weight_initialization.txt", "w+")

    topics = lda.show_topics(formatted = False, num_topics=lda.num_topics, num_words=len(dictionary))
    for topic in topics:
        topic.sort(key=lambda tup: int(tup[1]))
        string = ""
        for word in topic:
            string += str(word[0]) + " "
        f.write(string + "\n")

    f.close()

def get_sentence(s):
    return "".join(s.split("|||")[1].split("|")[0:-1:2]).strip().replace("  ", " ").lower()

def get_sbleu_file(n_best_list_file_name, reference_file_name):

    f = open("data/sbleu.txt", "w+")
    
    n_best_list = LazyFileReader(n_best_list_file_name)
    reference_list = LazyFileReader(reference_file_name)
    iter_n_best = iter(n_best_list)

    hypothesis = iter_n_best.next()

    for n, reference in enumerate(reference_list):

        while (n == int(hypothesis.split("|||")[0].strip())):
            hypothesis = get_sentence(hypothesis)
            sBleu = sbleu.bleu(hypothesis, reference, 3) 
            f.write("{}\n".format(sBleu))

            try:
                hypothesis = iter_n_best.next()
            except StopIteration:
                break

    f.close()

def main(corpus_reference_file_name, corpus_target_file_name, n_best_list_file_name, reference_file_name):
    if not os.path.exists("data"):
        os.makedirs("data")

    corpus_reference = LazyFileReader(corpus_reference_file_name)
    corpus_target = LazyFileReader(corpus_target_file_name)
    corpus_combined = MyCorpus(corpus_reference, corpus_target)

    print "Building dictionary .."
    corpus, dictionary = build_corpus_dictionary(corpus_combined)

    print "Running LDA .."
    lda = run_lda(corpus, 100)

    print "Saving topics to file .."
    save_topics(lda, dictionary)
    W1 = np.loadtxt('data/weight_initialization.txt').transpose()
    np.savetxt('data/weight_initialization.gz', W1)  

    print "Calucating sbleu scores .."
    get_sbleu_file(n_best_list_file_name, reference_file_name)  

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
