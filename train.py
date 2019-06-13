from collections import Counter
import pickle
import string


def messageFunction(message):
    # TODO: Add lemmatization and stemmer
    with open('stopwords-en.txt',encoding="utf8") as f:
        stopwords = {i.strip() for i in f.readlines()}

    message=''.join([i for i in message if i not in string.punctuation])
    message = message.lower().split()
    message = [word for word in message if word not in stopwords]
    return message

# Predicting
def predict(message):
    message = messageFunction(message)
    with open('probab.pickle', 'rb') as f:
        probablity = pickle.load(f)
    with open('word_count.pickle', 'rb') as f:
        word_counter = pickle.load(f)
    for word in message:
        for label, counter in word_counter.items():
            if counter.get(word, False):
                probablity[label] *= counter.get(word)
    return max(probablity, key=lambda x: probablity[x])

if __name__ == "__main__":
    # Preprocessing
    with open('SMSSpamCollection',encoding="utf8") as f:
        tmp = f.readlines()

    word_counter = {}
    count = {}
    total_count = 0

    # Training
    for i in tmp:
        i.strip()
        label, message = i.split("\t")
        message = messageFunction(message)
        word_counter[label] = word_counter.setdefault(
            label, Counter())+Counter(message)
        count[label] = count.setdefault(label, 0)+1
        total_count += 1

    # Saving Training output
    probablity = {k: v/total_count for k, v in count.items()}
    with open('probab.pickle', 'wb') as f:
        pickle.dump(probablity, f)
    with open('word_count.pickle', 'wb') as f:
        pickle.dump(word_counter, f)





    # Print score/accuracy
    right, total = 0, 0
    for i in tmp:
        i.strip()
        label, message = i.split("\t")
        if predict(message) == label:
            right += 1
        total += 1
    print((right, total), right/total)
