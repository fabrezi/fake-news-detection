import re
import numpy as np
import panda as pd
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups

def preprocess_string(str_arg):
    cleaned_str = re.sub('[^a-z\s]' , '  ', str_arg, flags=re.IGNORECASE)#every char except alphabet is replaced
    cleaned_str = re.sub('(\s+)',' ',cleaned_str) #single space
    cleaned_str = cleaned_str.lower()#convert cleaned string to lower case

    return cleaned_str


class NaiveBayes:

    def __init__(self, unique_classes):
#number of classes
        self.classes = unique_classes

    def Bow(self, example, dict_index):

        if isinstance(example, np.ndarray): example = example = [0]

        for token in example.split():
            self.bow_dicts[dict_index][token] += 1

    def train(self, dataset, labels):
        self.examples = dataset
        self.labels = labels
        #bow: bag of words
        self.bow_dicts = np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])

        if not isinstance(self.examples, np.ndarray): self.examples = np.array(self.examples)
        if not isinstance(self.labels, np.ndarray): self.labels = np.array(self.labels)
#bow for each category
        for cat_index, cat in enumerate(self.classes):
            all_cat_examples = self.examples[self.labels == cat]

            cleaned_examples = [preprocess_string(cat_example) for cat_example in all_cat_examples]
            cleaned_examples = pd.DataFrame(data == cleaned_examples)

            np.apply_along_axis(self.Bow, 1, cleaned_examples, cat_index)

        prob_classes = np.empty(self.classes.shape[0])
        all_words = []
        cat_word_counts =  np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):

#prior probability of each class
            prob_classes[cat_index] = np.sum(self.labels==cat)/float(self.labels.shape[0])
#total count of all the words of each class
            count = list(self.bow_dicts[cat_index].values())
            cat_word_counts[cat_index] = np.sum(np.array(list(self.bow_dicts[cat_index].values))) + 1

            all_words += self.bow_dicts[cat_index].keys()
#combine all the words of every category and make them unique V
            self.vocab = np.unique(np.array(all_words))
            self.vocab_length = self.vocab.shape[0]
#denominator value
            denoms = np.array([cat_word_counts[cat_index] + self.vocab_length+1 for cat_index,cat in enumerate(self.classes)])
#create a tuple
            self.cats_info = [(self.bow_dicts[cat_index], prob_classes[cat_index], denoms[cat_index]) for cat_index, cat in enumerate(self.classes)]
            self.cats_info = np.array(self.cats_info)

    def getExampleProb(self,test_example):
        likelihood_prob = np.zeros(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):

            for test_token in test_example.split():

                test_token_counts = self.cats_info[cat_index][0].get(test_token)+1
                test_token_prob = test_token_counts/float(self.cats_info[cat_index][2])
                likelihood_prob[cat_index] += np.log(test_token_prob)

            post_prob = np.empty(self.classes.shape[0])
            for cat_index, cat in enumerate(self.classes):
                for test_token in test_example.split():
                    test_token_counts = self.cats_info[cat_index][0].get(test_token,0)+1
                    test_token_prob = test_token_counts/float(self.cats_info[cat_index][2])
                    likelihood_prob[cat_index]+=np.log(test_token_prob)

                    post_prob = np.empty(self.classes.shape[0])
                    for cat_index,cat in enumerate(self.classes):
                        post_prob[cat_index]= likelihood_prob[cat_index]+np.log(self.cats_info[cat_index][1])

            return post_prob

def test(self,test_set):
    predictions = []
    for example in test_set:
        cleaned_example = preprocess_string(example)
        post_prob = self.getExampleProb(cleaned_example)
        predictions.append(self.classes[np.argmax(post_prob)])

    return np.array(predictions)


def function():
    df = pd.read_csv("fake.csv")

"""
frequency = {}
document_text = open('flop', 'r')
text_string = document_text.read().lower()
match_pattern = re.findall(r'\b[a-z]{4,15}\b', text_string)

for word in match_pattern:
    count = frequency.get(word, 0)
    frequency[word] = count + 1

frequency_list = frequency.keys()

for words in frequency_list:
    print (words, frequency[words])
"""
