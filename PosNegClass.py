import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model
import random
import nltk
 # -*- coding: utf-8 -*-


class PosNeg:

    try:

        model1 = load_model('model_bot.h5')
        s = "لا ماعندي سعال جاف "

        s2 = "لا ما احس بتعب"

        s3 = "نعم احس بتعب"

        intents = json.loads(open('intents.json', encoding="utf-8").read())
        words = pickle.load(open('words.pkl', 'rb', ))
        classes = pickle.load(open('classes.pkl', 'rb', ))

        def clean_up_sentence(self, sentence):
            sentence_words = nltk.word_tokenize(sentence)
            return sentence_words

        def bow(self, sentence, words, show_details=True):
            sentence_words = self.clean_up_sentence(sentence)
            bag = [0] * len(words)
            for s in sentence_words:
                for i, w in enumerate(words):
                    if w == s:
                        bag[i] = 1
                        if show_details:
                            print("found in bag: %s" % w)
            return (numpy.array(bag))

        def predict_class(self, sentence, model=model1):
            rValue = True
            p = self.bow(sentence, self.words, show_details=False)
            res = model.predict(numpy.array([p]))[0]
            ERROR_THRESHOLD = 0.25
            results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = []
            for r in results:
                return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
            print(return_list)
            if self.classes[r[0]] == "pos":
                rValue = True
            else:
                rValue = False
            return rValue

        print("FINAL RESULTS s : ")
        print(predict_class(s, model1))

        print("FINAL RESULTS s2 : ")
        print(predict_class(s2, model1))

        print("FINAL RESULTS s3 : ")
        print(predict_class(s3, model1))
    except:

        words = []
        classes = []
        documents = []

        data_file = open('intents.json', encoding="utf-8").read()
        intents = json.loads(data_file)

        # Preprocess data

        for intent in intents['intents']:
            for pattern in intent['patterns']:
                w = nltk.word_tokenize(pattern)
                words.extend(w)
                documents.append((w, intent['tag']))
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))
        print(len(documents), "documents", documents)
        print(len(classes), "classes", classes)
        print(len(words), "lemmatized words", words)

        pickle.dump(words, open('words.pkl', 'wb'))
        pickle.dump(classes, open('classes.pkl', 'wb'))

        # training
        training = []
        output_empty = [0] * len(classes)
        for doc in documents:
            bag = []
            pattern_words = doc[0]

            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        print("Done training data")

        # Build model
        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd_o = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd_o, metrics=['accuracy'])

        hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        model.save('model_bot.h5', hist)
        print("Done model")