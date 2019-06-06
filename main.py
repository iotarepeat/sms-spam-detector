import pickle


def messageFunction(message):
    # TODO: Add lemmatization and stemmer
    with open('stopwords-en.txt') as f:
        stopwords = {i.strip() for i in f.readlines()}
    for char in '!@$&.%#':
        message.replace(char, '')
    message = message.lower().split()
    message = [word for word in message if word not in stopwords]
    return message


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


message = input("Enter txt message:")
print("Message is most likely: ", predict(message))
