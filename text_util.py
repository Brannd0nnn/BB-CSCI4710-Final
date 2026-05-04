import re

def clean_text(text):
    #Convert all text into lowercase words so its treated equally
    text = text.lower()
    #Removes punctuation, numbers, and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    #Split cleaned text into individual words
    return text.split()