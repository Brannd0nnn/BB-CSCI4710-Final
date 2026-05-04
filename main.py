import math
from graph import show_probability_graph
from classifier import NaiveBayesSpamClassifier
from data import training_data

def main():
    classifier = NaiveBayesSpamClassifier()
    classifier.train(training_data)

    print("\nSpam Message Classifier\n")

    print("Type a message to check if it is spam.\n")
    print("Type 'quit' to exit.\n")

    while True:
        user_message = input("Enter a message: ")

        if user_message.lower() == "quit":
            print("Program Ending...")
            break

        # Predict whether the message is spam or not spam
        # Also returns overall confidence percentage
        prediction, confidence = classifier.predict(user_message)

        # Calculate the raw logarithmic probability score for spam
        spam_score = classifier.calculate_probability(user_message, "spam")

        # Calculate the raw logarithmic probability score for not spam
        not_spam_score = classifier.calculate_probability(user_message, "not spam")

        #Convert scores back into normal probabilities
        spam_prob = math.exp(spam_score)
        not_spam_prob = math.exp(not_spam_score)

        # Add both probabilities together for normalization
        total_prob = spam_prob + not_spam_prob

        # Convert spam and not spam probability into percentage form
        spam_percentage = (spam_prob / total_prob) * 100
        not_spam_percentage = (not_spam_prob / total_prob) * 100

        print("\nRESULTS")
        print(f"Message: {user_message}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Spam Probability: {spam_percentage:.2f}%")
        print(f"Not Spam Probability: {not_spam_percentage:.2f}%\n")

        #Display NetworkX graph with user input, spam probability, and not spam probability
        show_probability_graph(
            user_message,
            spam_percentage,
            not_spam_percentage
        )

if __name__ == "__main__":
    main()