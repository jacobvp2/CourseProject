import numpy as np
import math
from collections import Counter
import time
import random
from generateData import topic1,topic2

e_step = 2
def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix


class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.word_matrix = []

        self.documents_path = documents_path

        self.term_doc_matrix = None
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0
        self.alpha = 5
        self.gradient = ''

        self.max_iterations = 11


    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]

        Update self.number_of_documents
        """
        list_of_list = []

        document = open(self.documents_path, 'r')
        r_lines = document.readlines()

        for l in range(len(r_lines)):
            split_by_line = r_lines[l].split("\n")
            split_by_word = split_by_line[0].split(" ")
            list_of_list.append(split_by_word)

        self.documents = list_of_list
        self.number_of_documents = len(list_of_list)

        # pass    # REMOVE THIS

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """

        list_of_all_uwords = []
        for doc in self.documents:
            for w in doc:
                if(w not in list_of_all_uwords):
                    list_of_all_uwords.append(w)

        self.vocabulary = list_of_all_uwords
        self.vocabulary_size = len(list_of_all_uwords)
        # pass    # REMOVE THIS

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document,
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        term_doc_matrix = []
        word_matrix = []

        for document in self.documents:
            word_counts = Counter()
            for word in document:
                word_counts[word] += 1

            document_row = []
            word_row = []
            for word in self.vocabulary:
                document_row.append(word_counts[word])
                word_row.append(word)

            term_doc_matrix.append(document_row)
            word_matrix.append(word_row)

        self.term_doc_matrix = np.array(term_doc_matrix)
        self.word_matrix = word_matrix


    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize!
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """

        self.document_topic_prob = np.random.random_sample((self.number_of_documents, number_of_topics))
        self.topic_word_prob = np.random.random_sample((number_of_topics, len(self.vocabulary)))

        self.document_topic_prob = normalize(self.document_topic_prob)
        self.topic_word_prob = normalize(self.topic_word_prob)


    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """

        for i in range(self.number_of_documents):
            for j in range(e_step):
                for k in range(self.vocabulary_size):
                    self.topic_prob[i, j, k] = self.topic_word_prob[j, k] * self.document_topic_prob[i, j]

        for i in range(self.number_of_documents):
            for j in range(self.vocabulary_size):
                norm_constant = 0.0
                for k in range(e_step):
                    norm_constant += self.topic_prob[i, k, j]
                for k in range(e_step):
                    self.topic_prob[i, k, j] = self.topic_prob[i, k, j] / norm_constant



    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        for i in range(number_of_topics):
            for j in range(self.vocabulary_size):
                ct = 0.0
                for k in range(self.number_of_documents):
                    ct += (self.term_doc_matrix[k][j] * self.topic_prob[k, i, j])
                self.topic_word_prob[i, j] = ct

        for i in range(number_of_topics):
            norm_constant = 0.0
            for j in range(self.vocabulary_size):
                norm_constant += self.topic_word_prob[i, j]
            for j_done in range(self.vocabulary_size):
                self.topic_word_prob[i, j_done] += (self.topic_word_prob[i, j_done] / norm_constant)

        for i in range(self.number_of_documents):
            for j in range(number_of_topics):
                ct = 0.0
                for k in range(self.vocabulary_size):
                    ct += (self.term_doc_matrix[i][k] * self.topic_prob[i, j, k])
                self.document_topic_prob[i, j] = ct

        for i in range(self.number_of_documents):
            norm_constant = 0.0
            for j in range(number_of_topics):
                norm_constant += self.document_topic_prob[i, j]
            for j_done in range(number_of_topics):
                self.document_topic_prob[i, j_done] = (self.document_topic_prob[i, j_done] / norm_constant)


    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices

        Append the calculated log-likelihood to self.likelihoods

        """

        log_liklihood = 1.0
        for i in range(self.number_of_documents):
            for j in range(self.vocabulary_size):
                top_total = 0.01
                for k in range(number_of_topics):
                    cond_prob = self.topic_word_prob[k, j] * self.document_topic_prob[i, k]
                    top_total += cond_prob

                log_comp = self.term_doc_matrix[i][j] * np.log(top_total)
                log_liklihood += log_comp
        self.likelihoods.append(log_liklihood)

        return log_liklihood



    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")

        # build term-doc matrix
        self.build_term_doc_matrix()

        # Create the counter arrays.

        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        initial_origin = 0
        # self.num_test = number_of_topics
        global e_step
        e_step = number_of_topics
        for iteration in range(self.max_iterations):
            print("Iteration #" + str(iteration + 1) + "...")

            self.expectation_step()
            self.maximization_step(number_of_topics)
            self.calculate_likelihood(number_of_topics)

            if (abs(self.topic_prob[initial_origin][initial_origin][initial_origin] - 1.00) < epsilon):
                break


        # print(top_term_matrix[:5])
        # print(self.term_doc_matrix)

        freq_map = {}

        for i in range(self.number_of_documents):
            for j in range(self.vocabulary_size):
                if self.term_doc_matrix[i][j] not in freq_map:
                    freq_map[self.term_doc_matrix[i][j]] = []

                freq_map[self.term_doc_matrix[i][j]].append(self.word_matrix[i][j])

        keys = sorted([key for key in freq_map], reverse=True)

        pairs = []
        while len(pairs) < 5:
            for count in keys:
                for term in freq_map[count]:
                    pairs.append((term, count))

        s = open('data/topic1.txt', 'r')
        text = s.read().split()
        text = [''.join(filter(str.isalnum, input.lower())) for input in text]
        s2 = open('data/topic2.txt', 'r')
        text2 = s2.read().split()
        text2 = [''.join(filter(str.isalnum, input.lower())) for input in text2]

        stop_words = set("i me my myself we our youre ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why us said how all any both each few more most other some such no nor not only own same so than too very s t can will just don should now".split())

        stop_words.add(topic1.lower())
        stop_words.add(topic2.lower())

        counts = Counter(text)
        counts2 = Counter(text2)

        top_words = []
        for word, _ in counts.most_common():
            if word in counts2 and word not in stop_words and word is not self.gradient and len(word) > self.alpha:
                top_words.append(word)

                if len(top_words) == self.alpha:
                    break
        print(top_words)
        f = open("theme.txt","w+")
        f.write("Top Common Themes Are: {}".format(', '.join(top_words)))
        f.close()

def main():
    data_repo_paths = ['data/topic1.txt','data/topic2.txt']
    for document_path in data_repo_paths:
        documents_path = document_path

    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    # print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 10
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
