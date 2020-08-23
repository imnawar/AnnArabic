import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.isri import ISRIStemmer
stemmer = LancasterStemmer()
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model
# -*- coding: utf-8 -*-
#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import os, sys
from bot import telegram_chatbot
import numpy
import tensorflow
import random
import json
import tflearn
import pickle

try:

    model1 = load_model('model_bot.h5')
    s = "لا ماعندي سعال جاف "

    s2 = "لا ما احس بتعب"

    s3 = "نعم احس بتعب"

    intents1 = json.loads(open('intentsPosNeg.json', encoding="utf-8").read())
    words1 = pickle.load(open('words.pkl', 'rb', ))
    classes = pickle.load(open('classes.pkl', 'rb', ))


    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        return sentence_words


    def bow(sentence, words1, show_details=True):
        sentence_words = clean_up_sentence(sentence)
        bag1 = [0] * len(words1)
        for s in sentence_words:
            for i, w in enumerate(words1):
                if w == s:
                    bag1[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return (numpy.array(bag1))


    def predict_class(sentence, model=model1):
        rValue = True
        p = bow(sentence, words1, show_details=False)
        res = model.predict(numpy.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        print(return_list)
        if classes[r[0]] == "pos":
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

    words1 = []
    classes = []
    documents = []

    data_file = open('intentsPosNeg.json', encoding="utf-8").read()
    intents1 = json.loads(data_file)

    # Preprocess data

    for intent in intents1['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words1.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words1 = sorted(list(set(words1)))
    classes = sorted(list(set(classes)))
    print(len(documents), "documents", documents)
    print(len(classes), "classes", classes)
    print(len(words1), "lemmatized words", words1)

    pickle.dump(words1, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    # training
    training = []
    output_empty1 = [0] * len(classes)
    for doc in documents:
        bag = []
        pattern_words = doc[0]

        for w in words1:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty1)
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


def stem(w):
    isri_stemmer = ISRIStemmer()
    return isri_stemmer.stem(w)


with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb", encoding="utf-8") as f:
        words, labels, traning, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stem(w) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    traning = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        traning.append(bag)
        output.append(output_row)

    traning = numpy.array(traning)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, traning, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(traning[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)

net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(traning, output, n_epoch=300, batch_size=2, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stem(w) for w in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


bot = telegram_chatbot("config.cfg")

def stetmentAnalyisi(s):
    rv = predict_class(s)
    print("PREDDDDDDIIIIIICCCTTTT: ")
    print(rv)
    return rv


def chat():
    startMassage ='ياهلا والله ومرحبا\nاعرفكم بنفسي انا آن المساعد موجود هنا عشان اساعدكم واعطيكم معلومات عن كورونا عافانا الله واياكم،\n-لمعرفة ماهو فايروس كورونا:\n     "ماهو فايروس كورونا؟"\n-لمعرفة اعراض الفايروس:\n     "ماهي اعراض فايروس كورونا؟"\n-لمعرفة طرق الوقايه من فايروس كورونا:\n     "كيف احمي نفسي من كورونا؟"\n-ولحساب احتمال نسبة اصابتكم بالمرض لا قدر الله:\n     "اعتقد عندي كورونا"\nوبعدها جاوبني على الاسئلة بشكل واضح وكامل\nوللدعم الفني:\nmanalnawar@outlook.sa\دمتم بود  '
    isymptoms = {"fever": 0, "drycough": 0, "tiredness": 0, "sorethroat": 0, "diarrhoea": 0, "lossoftasteorsmell": 0,
                 "achesandpains": 0, "headache": 0, "conjunctivitis": 0}
    user_probability = {}
    sym_us = {}
    us_factor = {}
    update_id = None
    while True:
        updates = bot.get_updates(offset=update_id)
        updates = updates["result"]
        factor = 1
        if updates:
            for item in updates:
                update_id = item["update_id"]
                try:
                    from_ = item["message"]["from"]["id"]
                    if from_ not in user_probability:
                        user_probability[from_] = 0
                        sym_us[from_] = isymptoms.copy()
                        us_factor[from_] = 1
                        print("Users:")
                        print(len(user_probability))
                    inp = item["message"]["text"]
                    if inp in ["/start", "//أبدأ"]:
                        rep = startMassage
                    else:
                        results = model.predict([bag_of_words(inp, words)])
                        results_index = numpy.argmax(results)
                        tag = labels[results_index]
                        for tg in data["intents"]:
                            if tg['tag'] == tag:
                                t = tag
                                print("tag of input : ", t)
                                responses = tg['responses']
                                if (tag ==  "days" and not stetmentAnalyisi(inp)):
                                    us_factor[from_] = 0.5
                                if (tag in isymptoms and stetmentAnalyisi(inp)):
                                    sym_us[from_][t] = tg['probability']
                        if tag == "end":
                            for x in isymptoms:
                                user_probability[from_] = user_probability[from_] + float(sym_us[from_][x])
                            user_probability[from_] = user_probability[from_] * us_factor[from_]
                            user_probability[from_] = format(user_probability[from_], '.2f')
                            rep = random.choice(responses) + str(user_probability[from_]) + "%"
                        elif (tag == "goodbye"):
                            del sym_us[from_]
                            del user_probability[from_]
                            del us_factor[from_]
                            rep = random.choice(responses)
                        else:
                            rep = random.choice(responses)


                except:
                    message = None
                    rep = None
                bot.send_message(rep, from_)

chat()
