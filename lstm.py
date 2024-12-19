import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("gmb.csv", sep=",", encoding="latin1")
df["Word"] = df["Word"].astype(str)
df = df.dropna(subset=["Word", "Tag", "POS"])
#only used the first 8000 to reduce the waiting time
df = df.head(8000)

X_word = df["Word"]
y_pos = df["POS"]
y_tag = df["Tag"]

y_tag, tag_cl = pd.factorize(y_tag)
y_pos, pos_cl = pd.factorize(y_pos)

X_train, X_test, y_tag_train, y_tag_test, y_pos_train, y_pos_test = train_test_split(
    X_word, y_tag, y_pos, test_size=0.1, random_state=42)

X_train = X_train.tolist()
X_test = X_test.tolist()

#tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
tok_X_train = tokenizer.texts_to_sequences(X_train)
tok_X_test = tokenizer.texts_to_sequences(X_test)
tok_X_train = pad_sequences(tok_X_train, maxlen=MAX_LENGTH, padding="post")
tok_X_test = pad_sequences(tok_X_test, maxlen=MAX_LENGTH, padding="post")
vocab_size = len(tokenizer.word_index) + 1
MAX_LENGTH = max(len(tok_word) for tok_word in tok_X_train)

#LSTM
LSTM_POS = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LENGTH),
    LSTM(64, activation="tanh", recurrent_activation="sigmoid",
         dropout=0.2, recurrent_dropout=0.3,
         return_sequences=True,
         kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
         bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-4),
         recurrent_regularizer=L2(1e-4)),
    Flatten(),
    Dense(32, activation="tanh"),
    Dropout(0.4),
    Dense(len(pos_cl), activation="softmax", name="tag_output")
])

LSTM_tag = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LENGTH),
    LSTM(64, activation="tanh", recurrent_activation="sigmoid",
         dropout=0.2, recurrent_dropout=0.3,
         return_sequences=True,
         kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
         bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-4),
         recurrent_regularizer=L2(1e-4)),
    Flatten(),
    Dense(32, activation="tanh"),
    Dropout(0.4),
    Dense(len(tag_cl), activation="softmax", name="tag_output")
])

LSTM_POS.compile(optimizer=Adam(learning_rate=0.001), 
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

LSTM_tag.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

LSTM_POS.fit(tok_X_train, y_pos_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
LSTM_tag.fit(tok_X_train, y_tag_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

y_pred_pos = LSTM_POS.predict(tok_X_test)
y_pred_pos = np.argmax(y_pred_pos, axis=1)
y_pred_tag = LSTM_tag.predict(tok_X_test)
y_pred_tag = np.argmax(y_pred_tag, axis=1)
print("LSTM Model:")
print(classification_report(y_pos_test, y_pred_pos))
print(classification_report(y_tag_test, y_pred_tag))


#BiLSTM
BiLSTM_POS = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LENGTH),
    Bidirectional(LSTM(64, activation="tanh", recurrent_activation="sigmoid",
                       dropout=0.3, recurrent_dropout=0.3,
                       return_sequences=True,
                       kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                       bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-4),
                       recurrent_regularizer=L2(1e-4))),
    Flatten(),
    Dense(32, activation='tanh'),
    Dropout(0.4),
    Dense(len(pos_cl), activation='softmax', name="tag_output")
])

BiLSTM_TAG = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LENGTH),
    Bidirectional(LSTM(64, activation="tanh", recurrent_activation="sigmoid",
                       dropout=0.3, recurrent_dropout=0.3,
                       return_sequences=True,
                       kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                       bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-4),
                       recurrent_regularizer=L2(1e-4))),
    Flatten(),
    Dense(32, activation='tanh'),
    Dropout(0.4),
    Dense(len(tag_cl), activation='softmax', name="tag_output")
])

BiLSTM_POS.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

BiLSTM_TAG.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

BiLSTM_POS.fit(tok_X_train, y_pos_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
BiLSTM_TAG.fit(tok_X_train, y_tag_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

y_pred_pos = BiLSTM_POS.predict(tok_X_test)
y_pred_pos = np.argmax(y_pred_pos, axis=1)
y_pred_tag = BiLSTM_TAG.predict(tok_X_test)
y_pred_tag = np.argmax(y_pred_tag, axis=1)
print("BiLSTM Model:")
print(classification_report(y_pos_test, y_pred_pos))
print(classification_report(y_tag_test, y_pred_tag))


#RNN
RNN_POS = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LENGTH),
    SimpleRNN(64, activation="tanh", 
              dropout=0.3, recurrent_dropout=0.3, 
              return_sequences=True,
              kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
              bias_regularizer=L2(1e-4), 
              activity_regularizer=L2(1e-4),
              recurrent_regularizer=L2(1e-4)),
    Flatten(),
    Dense(32, activation='tanh'),
    Dropout(0.4),
    Dense(len(pos_cl), activation='softmax', name="tag_output")
])

RNN_TAG = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LENGTH),
    SimpleRNN(64, activation="tanh", 
              dropout=0.3, recurrent_dropout=0.3, 
              return_sequences=True,
              kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
              bias_regularizer=L2(1e-4), 
              activity_regularizer=L2(1e-4),
              recurrent_regularizer=L2(1e-4)),
    Flatten(),
    Dense(32, activation='tanh'),
    Dropout(0.4),
    Dense(len(tag_cl), activation='softmax', name="tag_output")
])


RNN_POS.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

RNN_TAG.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

RNN_POS.fit(tok_X_train, y_pos_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
RNN_TAG.fit(tok_X_train, y_tag_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

y_pred_pos = RNN_POS.predict(tok_X_test)
y_pred_pos = np.argmax(y_pred_pos, axis=1)
y_pred_tag = RNN_TAG.predict(tok_X_test)
y_pred_tag = np.argmax(y_pred_tag, axis=1)
print("RNN Model:")
print(classification_report(y_pos_test, y_pred_pos))
print(classification_report(y_tag_test, y_pred_tag))