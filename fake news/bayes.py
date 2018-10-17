#class: real or fake
#features: words
import re
import numpy as np
import panda as pd
from collections import defaultdict

def preprocess_string(str_arg):
    cleaned_str = re.sub('[^a-z\s]' , '  ', str_arg, flags=re.IGNORECASE)#every char except alphabet is replaced
    cleaned_str = re.sub('(\s+)',' ',cleaned_str) #single space
    cleaned_str = cleaned_str.lower()#convert cleaned string to lower case

    return cleaned_str


class NaiveBayes:
#constructor
    def __init__(self, unique_classes):
#number of classes
        self.classes = unique_classes
#bag of words
    def Bow(self, example, dict_index):

        if isinstance(example, np.ndarray): example = example = [0]

        for token in example.split():
            self.bow_dicts[dict_index][token] += 1

    def train(self, dataset, labels):
        self.examples = dataset
        self.labels = labels
        self.bow_dicts = np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])

        if not isinstance(self.examples, np.ndarray):
            self.examples = np.array(self.examples)
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)

#bow for each category
        for dog_index, dog in enumerate(self.classes):
            all_dog_examples = self.examples[self.labels == dog]

            cleaned_examples = [preprocess_string(dog_example) for dog_example in all_dog_examples]
            cleaned_examples = pd.DataFrame(data == cleaned_examples)

            np.apply_along_axis(self.Bow, 1, cleaned_examples, dog_index)

        prob_classes = np.empty(self.classes.shape[0])
        all_words = []
        dog_word_counts =  np.empty(self.classes.shape[0])
        for dog_index, dog in enumerate(self.classes):

#prior probability of each class
            prob_classes[dog_index] = np.sum(self.labels==dog)/float(self.labels.shape[0])
#total count of all the words of each class
            count = list(self.bow_dicts[dog_index].values())
            dog_word_counts[dog_index] = np.sum(np.array(list(self.bow_dicts[dog_index].values))) + 1

            all_words += self.bow_dicts[dog_index].keys()
#combine all the words of every category and make them unique V
            self.vocab = np.unique(np.array(all_words))
            self.vocab_length = self.vocab.shape[0]
#denominator value
            denoms = np.array([dog_word_counts[dog_index] + self.vocab_length+1 for dog_index,dog in enumerate(self.classes)])
#create a tuple
            self.dogs_info = [(self.bow_dicts[dog_index], prob_classes[dog_index], denoms[dog_index]) for dog_index, dog in enumerate(self.classes)]
            self.dogs_info = np.array(self.dogs_info)

    def getExampleProb(self,test_example):
        likelihood_prob = np.zeros(self.classes.shape[0])
        for dog_index, dog in enumerate(self.classes):

            for test_token in test_example.split():

                test_token_counts = self.dogs_info[dog_index][0].get(test_token)+1
                test_token_prob = test_token_counts/float(self.dogs_info[dog_index][2])
                likelihood_prob[dog_index] += np.log(test_token_prob)

            post_prob = np.empty(self.classes.shape[0])
            for dog_index, dog in enumerate(self.classes):
                for test_token in test_example.split():
                    test_token_counts = self.dogs_info[dog_index][0].get(test_token,0)+1
                    test_token_prob = test_token_counts/float(self.dogs_info[dog_index][2])
                    likelihood_prob[dog_index]+=np.log(test_token_prob)

                    post_prob = np.empty(self.classes.shape[0])
                    for dog_index,dog in enumerate(self.classes):
                        post_prob[dog_index]= likelihood_prob[dog_index]+np.log(self.dogs_info[dog_index][1])

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
