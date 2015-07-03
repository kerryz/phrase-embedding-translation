import sbleu, linecache
from lazy_file_reader import LazyFileReader


def get_source_sentence_words(source_sentence):
    return source_sentence.strip().split(" ");

def get_phrase_pair_dict_n(source_sentence, target_sentence):
    """
    Returns a dictionary with phrase pairs and its counts of the given source and target sentences.

    @source_sentence - source sentence with tokens/words separated by a whitespace
    @target_sentence - target sentence, directly from the n best list
    return - dictionary with phrase pair as key and its count as value
    """

    source_sentence_words = get_source_sentence_words(source_sentence)
    alignments = target_sentence.split("|||")[1].split("|")[1:-1:2]
    target_phrases = target_sentence.split("|||")[1].split("|")[0:-1:2]

    dictonary = {}

    for target_phrase, alignment in zip(target_phrases, alignments):
        alignment = alignment.split("-")
        source_phrase = " ".join(source_sentence_words[int(alignment[0]):int(alignment[1])+1])

        t = (source_phrase, target_phrase)

        if t in dictonary:
            dictonary[t] += 1
        else:
            dictonary[t] = 1

    return dictonary

def combineDictionaries(phrase_pair_dict_all, phrase_pair_dict_n):

    for phrase_pair, count in phrase_pair_dict_n.iteritems():
        if phrase_pair in phrase_pair_dict_all:
            phrase_pair_dict_all[phrase_pair] += count
        else:
            phrase_pair_dict_all[phrase_pair] = count

    return phrase_pair_dict_all

def get_base_total_score(target_sentence):
    return float(target_sentence.split("|||")[3].strip())

def get_n_best_list_sentence_index(target_sentence):
    return int(target_sentence.split("|||")[0].strip())

def get_phrase_pair_lists_and_dicts(source_sentence, n_best_list):
    """
    Returns a list of phrase pair dictonaries for a given source sentence and possible translation. Also a dictionary over all n best list translations.

    @source_sentence - source sentence with tokens/words separated by a whitespace
    @n_best_list - n best list of possible translations of source_sentence
    return - a list of phrase pair dictonaries for a given source sentence and possible translation. Also a dictionary over all n best list translations
    """

    phrase_pair_dict_n_list = []
    phrase_pair_dict_all = {}

    for target_sentence in n_best_list:

        phrase_pair_dict_n = {}

        source_sentence_words = get_source_sentence_words(source_sentence)
        alignments = target_sentence.split("|||")[1].split("|")[1:-1:2]
        target_phrases = target_sentence.split("|||")[1].split("|")[0:-1:2]

        for target_phrase, alignment in zip(target_phrases, alignments):
            alignment = alignment.split("-")
            source_phrase = " ".join(source_sentence_words[int(alignment[0]):int(alignment[1])+1])

            t = (source_phrase, target_phrase)

            if t in phrase_pair_dict_n:
                phrase_pair_dict_n[t] += 1
            else:
                phrase_pair_dict_n[t] = 1

            if t in phrase_pair_dict_all:
                phrase_pair_dict_all[t] += 1
            else:
                phrase_pair_dict_all[t] = 1            

        phrase_pair_dict_n_list.append(phrase_pair_dict_n)

    return phrase_pair_dict_n_list, phrase_pair_dict_all

def get_n_best_list_sbleu_score_list_and_total_base_score_list(source_sentence_index, start_line_n_best_list_list, sbleu_score_list_file_name, n_best_list_file_name):
    """
    Returns a list with n best translations of a source_sentence, a sentence-level BLEU score list for these translations and a list with the total feature value of these translations.

    @source_sentence_index - index for a source sentence
    @start_line_n_best_list_list - list of lines in the n best list that should be fetched 
    @sbleu_score_list_file_name - path to file of the pre-computed sbleu score for all n best list sentences.
    @n_best_list_file_name - path to n_best_list file
    return - a list with n best translations of a source_sentence, a sentence-level BLEU score list for these translations and a list with the total feature value of these translations
    """    

    start_line_index = start_line_n_best_list_list[source_sentence_index]
    stop_line_index = start_line_n_best_list_list[source_sentence_index+1]

    n_best_list = []
    sbleu_score_list = []
    total_base_score_list = []

    for line_index in xrange(start_line_index,stop_line_index):
        target_sentence = linecache.getline(n_best_list_file_name, line_index).strip().lower()

        n_best_list.append(target_sentence)

        total_base_score = get_base_total_score(target_sentence)
        total_base_score_list.append(total_base_score)

        sbleu_score = float(linecache.getline(sbleu_score_list_file_name, line_index).strip())
        sbleu_score_list.append(sbleu_score)

    return n_best_list, total_base_score_list, sbleu_score_list

def get_start_line_n_best_list_list(n_best_list_file_name):
    """
    Returns a list of start line indices where the n best list given a source sentence should be fetched.

    @n_best_list_file_name - path to n_best_list file
    return - a list of start line indices where the n best list given a source sentence should be fetched

    """
    start_line_n_best_list_list = []
    n_best_list = LazyFileReader(n_best_list_file_name)
    last_index = -1

    for i, target_sentence in enumerate(n_best_list):
        source_sentence_index = get_n_best_list_sentence_index(target_sentence)
        if (source_sentence_index != last_index):
            start_line_n_best_list_list.append(i+1)
            last_index = source_sentence_index

    start_line_n_best_list_list.append(i+1)
    return start_line_n_best_list_list

def get_everything(source_sentence_index, source_sentence_list_file_name, n_best_list_file_name, sbleu_score_list_file_name):
    """
    Returns a list values and parameters needed for the neural network.

    @source_sentence_index - index for a source sentence
    @source_sentence_list_file_name - path to the source_sentence file
    @n_best_list_file_name - path to n_best_list file
    @sbleu_score_list_file_name - path to the sbleu file
    return - phrase_pair_dict_all, phrase_pair_dict_n_list, total_base_score_list, sbleu_score_list

    """
    source_sentence_list = LazyFileReader(source_sentence_list_file_name)

    start_line_n_best_list_list = get_start_line_n_best_list_list(n_best_list_file_name)

    source_sentence = linecache.getline(source_sentence_list_file_name, source_sentence_index+1)

    n_best_list, total_base_score_list, sbleu_score_list = get_n_best_list_sbleu_score_list_and_total_base_score_list(source_sentence_index, start_line_n_best_list_list, sbleu_score_list_file_name, n_best_list_file_name)
    phrase_pair_dict_n_list, phrase_pair_dict_all = get_phrase_pair_lists_and_dicts(source_sentence, n_best_list)

    return phrase_pair_dict_all, phrase_pair_dict_n_list, total_base_score_list, sbleu_score_list
