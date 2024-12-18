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

df = pd.read_csv("gmb.csv", sep=",", encoding="latin1")
df = df.dropna(subset=["Word","POS","Tag"])
df["Word"] = df["Word"].astype(str)

X = df["Word"]
y1 = df["POS"]
y2 = df["Tag"]

y1, y1c = pd.factorize(y1)
y2, y2c = pd.factorize(y2)
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.1, random_state=41)

X_train = X_train.tolist()
X_test = X_test.tolist()

tokenizer = Tokenizer(char_level=True, lower=False)
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1  # +1 to account for the padding token
tokenized_X_train = tokenizer.texts_to_sequences(X_train)
tokenized_X_test = tokenizer.texts_to_sequences(X_test)
MAX_LENGTH = max(len(tokenized_text) for tokenized_text in tokenized_X_train)
tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding="post")
tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding="post")

regularizer = L1L2(l1=1e-5, l2=1e-4)

def rnn_model(vocab_size, MAX_LENGTH, y1c, y2c):
    model_rnn_y1 = Sequential()
    model_rnn_y1 .add(Input(shape=(MAX_LENGTH,)))
    model_rnn_y1 .add(Embedding(input_dim=vocab_size, output_dim=25, input_length=MAX_LENGTH))
    model_rnn_y1 .add(SimpleRNN(64, activation="relu", dropout=0.25, kernel_regularizer = regularizer, bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5), return_sequences = True, recurrent_dropout = 0.25))
    model_rnn_y1 .add(Flatten())
    model_rnn_y1 .add(Dense(64, activation="relu"))
    model_rnn_y1 .add(Dropout(0.25))
    model_rnn_y1 .add(Dense(len(y1c), activation="relu"))
    model_rnn_y1 .compile(loss="crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=0.001))

    model_rnn_y2 = Sequential()
    model_rnn_y2 .add(Input(shape=(MAX_LENGTH,)))
    model_rnn_y2 .add(Embedding(input_dim=vocab_size, output_dim=25, input_length=MAX_LENGTH))
    model_rnn_y2 .add(SimpleRNN(64, activation="relu", dropout=0.25, kernel_regularizer = regularizer, bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5), return_sequences = True, recurrent_dropout = 0.25))
    model_rnn_y2 .add(Flatten())
    model_rnn_y2 .add(Dense(64, activation="relu"))
    model_rnn_y2 .add(Dropout(0.25))
    model_rnn_y2 .add(Dense(len(y2c), activation="relu"))
    model_rnn_y2 .compile(loss="crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=0.001))

    return model_rnn_y1, model_rnn_y2

def lstm_model(vocab_size, MAX_LENGTH, y1c, y2c):
    model_lstm_y1 = Sequential()
    model_lstm_y1 .add(Input(shape=(MAX_LENGTH,)))
    model_lstm_y1 .add(Embedding(input_dim=vocab_size, output_dim=25, input_length=MAX_LENGTH))
    model_lstm_y1 .add(LSTM(64, activation="relu", dropout=0.25, kernel_regularizer = regularizer, bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5), return_sequences = True, recurrent_dropout = 0.25))
    model_lstm_y1 .add(Flatten())
    model_lstm_y1 .add(Dense(64, activation="relu"))
    model_lstm_y1 .add(Dropout(0.25))
    model_lstm_y1 .add(Dense(len(y1c), activation="relu"))
    model_lstm_y1 .compile(loss="crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=0.001))
    
    model_lstm_y2 = Sequential()
    model_lstm_y2 .add(Input(shape=(MAX_LENGTH,)))
    model_lstm_y2 .add(Embedding(input_dim=vocab_size, output_dim=25, input_length=MAX_LENGTH))
    model_lstm_y2 .add(LSTM(64, activation="relu", dropout=0.25, kernel_regularizer = regularizer, bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5), return_sequences = True, recurrent_dropout = 0.25))
    model_lstm_y2 .add(Flatten())
    model_lstm_y2 .add(Dense(64, activation="relu"))
    model_lstm_y2 .add(Dropout(0.25))
    model_lstm_y2 .add(Dense(len(y2c), activation="relu"))
    model_lstm_y2 .compile(loss="crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=0.001))
    
    return model_lstm_y1, model_lstm_y2

def bilstm_model(vocab_size, MAX_LENGTH, y1c, y2c):
    model_bilstm_y1 = Sequential()
    model_bilstm_y1 .add(Input(shape=(MAX_LENGTH,)))
    model_bilstm_y1 .add(Embedding(input_dim=vocab_size, output_dim=25, input_length=MAX_LENGTH))
    model_bilstm_y1 .add(Bidirectional(LSTM(64, activation="relu", dropout=0.25, kernel_regularizer = regularizer, bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5), return_sequences = True, recurrent_dropout = 0.25)))
    model_bilstm_y1 .add(Flatten())
    model_bilstm_y1 .add(Dense(64, activation="relu"))
    model_bilstm_y1 .add(Dropout(0.25))
    model_bilstm_y1 .add(Dense(len(y1c), activation="relu"))
    model_bilstm_y1 .compile(loss="crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=0.001))
    
    model_bilstm_y2 = Sequential()
    model_bilstm_y2 .add(Input(shape=(MAX_LENGTH,)))
    model_bilstm_y2 .add(Embedding(input_dim=vocab_size, output_dim=25, input_length=MAX_LENGTH))
    model_bilstm_y2 .add(Bidirectional(LSTM(64, activation="relu", dropout=0.25, kernel_regularizer = regularizer, bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5), return_sequences = True, recurrent_dropout = 0.25)))
    model_bilstm_y2 .add(Flatten())
    model_bilstm_y2 .add(Dense(64, activation="relu"))
    model_bilstm_y2 .add(Dropout(0.25))
    model_bilstm_y2 .add(Dense(len(y2c), activation="relu"))
    model_bilstm_y2 .compile(loss="crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=0.001))
    
    return model_bilstm_y1, model_bilstm_y2

print("Welches Modell? rnn, lstm or bilstm")
chosenmodel = input()
if chosenmodel == "rnn":
    model_y1, model_y2 = rnn_model(vocab_size, MAX_LENGTH, y1c, y2c)
elif chosenmodel == "lstm":
    model_y1, model_y2 = lstm_model(vocab_size, MAX_LENGTH, y1c, y2c)
elif chosenmodel == "bilstm":
    model_y1, model_y2 = bilstm_model(vocab_size, MAX_LENGTH, y1c, y2c)
else:
    print("Error")

model_y1.fit(tokenized_X_train, y1_train, epochs = 5, batch_size = 64, validation_split = 0.1, verbose = 1)
model_y2.fit(tokenized_X_train, y2_train, epochs = 5, batch_size = 64, validation_split = 0.1, verbose = 1)

y1_pred = model_y1.predict(tokenized_X_test)
y1_pred = y1_pred.argmax(axis=1)

y2_pred = model_y2.predict(tokenized_X_test)
y2_pred = y2_pred.argmax(axis = 1)

print("y1 report")
print(classification_report(y1_test, y1_pred))

print("y2 report")
print(classification_report(y2_test, y2_pred))
