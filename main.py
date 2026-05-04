import re
import math
from collections import defaultdict

# Training data: message, label
training_data = [
    ("Win money now", "spam"),
    ("Claim your free prize today", "spam"),
    ("Congratulations you won a gift card", "spam"),
    ("Click here to claim your reward", "spam"),
    ("Limited time offer buy now", "spam"),
    ("You have won cash", "spam"),

    ("Hey are we still meeting today", "not spam"),
    ("Can you send me the homework", "not spam"),
    ("I will call you later", "not spam"),
    ("Let me know when you get home", "not spam"),
    ("Are you coming to class tomorrow", "not spam"),
    ("Don’t forget about the project deadline", "not spam"),
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.split()

class NaiveBayesSpamClassifier:
    def __init__(self):
        self.word_counts = {
            "spam": defaultdict(int),
            "not spam": defaultdict(int)
        }
        self.class_counts = {
            "spam": 0,
            "not spam": 0
        }
        self.total_words = {
            "spam": 0,
            "not spam": 0
        }
        self.vocabulary = set()

    def train(self, data):
        for message, label in data:
            words = clean_text(message)
            self.class_counts[label] += 1

            for word in words:
                self.word_counts[label][word] += 1
                self.total_words[label] += 1
                self.vocabulary.add(word)

    def calculate_probability(self, message, label):
        words = clean_text(message)

        total_messages = sum(self.class_counts.values())
        class_probability = self.class_counts[label] / total_messages

        log_probability = math.log(class_probability)

        vocabulary_size = len(self.vocabulary)

        for word in words:
            word_count = self.word_counts[label][word]


            word_probability = (word_count + 1) / (
                self.total_words[label] + vocabulary_size
            )

            log_probability += math.log(word_probability)

        return log_probability

    def predict(self, message):
        spam_score = self.calculate_probability(message, "spam")
        not_spam_score = self.calculate_probability(message, "not spam")

        if spam_score > not_spam_score:
            prediction = "spam"
        else:
            prediction = "not spam"

        # Convert scores into confidence percentage
        spam_exp = math.exp(spam_score)
        not_spam_exp = math.exp(not_spam_score)
        total = spam_exp + not_spam_exp

        if prediction == "spam":
            confidence = spam_exp / total
        else:
            confidence = not_spam_exp / total

        return prediction, confidence * 100

def main():
    classifier = NaiveBayesSpamClassifier()
    classifier.train(training_data)

    print("Spam Message Classifier")
    print("-----------------------")
    print("Type a message to check if it is spam.")
    print("Type 'quit' to exit.\n")

    while True:
        user_message = input("Enter a message: ")

        if user_message.lower() == "quit":
            print("Goodbye!")
            break

        prediction, confidence = classifier.predict(user_message)

        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {confidence:.2f}%\n")

main()