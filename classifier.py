import math
from collections import defaultdict
from text_util import clean_text

class NaiveBayesSpamClassifier:
    #Constructor to initialize all data structures for runtime
    def __init__(self):
        #stores how many times each word appears in spam and not in spam
        self.word_counts = {
            "spam": defaultdict(int),
            "not spam": defaultdict(int)
        }
        #Tracks the total messages that belong in each class
        self.class_counts = {
            "spam": 0,
            "not spam": 0
        }
        #Tracks the total number of words in each class
        self.total_words = {
            "spam": 0,
            "not spam": 0
        }
        #Stores the unique words seen during runtime
        self.vocabulary = set()

    #Train classifier using labeled training data
    def train(self, data):
        #Loop each message and its label
        for message, label in data:
            #Clean and split the message into words
            words = clean_text(message)
            #Increase the message count for this class
            self.class_counts[label] += 1

            #Count each word
            for word in words:
                #Increase count of this word for label
                self.word_counts[label][word] += 1
                #Increase total word count for label
                self.total_words[label] += 1
                #Add word to vocabulary
                self.vocabulary.add(word)

    #Calculate probability score for message belonging to the specific class
    def calculate_probability(self, message, label):
        #Clean text using text utility
        words = clean_text(message)
        #Total messages during runtime
        total_messages = sum(self.class_counts.values())
        #Prior probability of this class
        class_probability = self.class_counts[label] / total_messages
        #Using logarithms to prevent float point underflow
        log_probability = math.log(class_probability)
        #Number of unique words
        vocabulary_size = len(self.vocabulary)
        #Calculate probability for each word
        for word in words:
            #Count how often word appears in this class
            word_count = self.word_counts[label][word]
            #Apply laplance smoothing
            word_probability = (word_count + 1) / (
                self.total_words[label] + vocabulary_size
            )
            #Add log probability
            log_probability += math.log(word_probability)

        return log_probability
    #Predict whether a message is spam or not spam
    def predict(self, message):
        #Calculate spam score
        spam_score = self.calculate_probability(message, "spam")
        #Calculate not spam score
        not_spam_score = self.calculate_probability(message, "not spam")

        #Chooses class with higher probability
        if spam_score > not_spam_score:
            prediction = "spam"
        else:
            prediction = "not spam"
        #Convert log probabilities back to normal numbers
        spam_exp = math.exp(spam_score)
        not_spam_exp = math.exp(not_spam_score)
        #Normalize total probability
        total = spam_exp + not_spam_exp
        #Determines confidence level
        if prediction == "spam":
            confidence = spam_exp / total
        else:
            confidence = not_spam_exp / total
        #Returns prediction of confidence percentage
        return prediction, confidence * 100