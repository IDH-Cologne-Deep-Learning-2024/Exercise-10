import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def to_number(labels):
    map_dict = {}
    count = 0
    for label in labels:
        if label not in map_dict.keys():
            map_dict[label] = count
            count += 1
    return [map_dict[x] for x in labels]


df = pd.read_csv("gmb.csv", sep=",", encoding='unicode_escape')
df = df.head(100_000)
df = df.dropna()

X = df.Word.astype(str).tolist()
y_pos = df.POS.tolist()
y_ner = df.Tag.tolist()
tasks = [y_pos, y_ner]

word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X)
X = word_tokenizer.texts_to_sequences(X)
vocab_size = len(word_tokenizer.word_index)+1

MAX_LENGTH = 1
X = pad_sequences(X, maxlen=MAX_LENGTH)

evaluation_scores = {}
model_types = ["LSTM", "RNN"]
for model_type in model_types:
    for task_name, y in zip(["POS", "NER"], tasks):
        number_classes = len(set(y))
        y = np.array(to_number(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=300, input_length=MAX_LENGTH))
        if model_type == "LSTM":
            model.add(Bidirectional(LSTM(64)))
        elif model_type == "RNN":
            model.add(Bidirectional(SimpleRNN(64)))
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss='crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=20, verbose=0)
        y_pred = model.predict(X_test)
        y_pred = y_pred.argmax(axis=1)
        evaluation_scores[f"{model_type}_{task_name}"] = classification_report(y_test, y_pred)

for k in evaluation_scores:
    print(k)
    print(evaluation_scores[k])
