import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2, L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load and preprocess data
df = pd.read_csv("gmb.csv", sep=",", encoding="latin1")
df = df.dropna(subset=["Word", "POS", "Tag"])
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
vocab_size = len(tokenizer.word_index) + 1
tokenized_X_train = tokenizer.texts_to_sequences(X_train)
tokenized_X_test = tokenizer.texts_to_sequences(X_test)
MAX_LENGTH = max(len(tokenized_text) for tokenized_text in tokenized_X_train)
tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding="post")
tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding="post")

regularizer = L1L2(l1=1e-5, l2=1e-4)

def create_model(model_type, vocab_size, MAX_LENGTH, output_dim):
    model = Sequential()
    model.add(Input(shape=(MAX_LENGTH,)))
    model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=MAX_LENGTH))
    
    if model_type == "rnn":
        model.add(SimpleRNN(64, activation="relu", dropout=0.25, kernel_regularizer=regularizer, bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5), return_sequences=True, recurrent_dropout=0.25))
    elif model_type == "lstm":
        model.add(LSTM(64, activation="relu", dropout=0.25, kernel_regularizer=regularizer, bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5), return_sequences=True, recurrent_dropout=0.25))
    elif model_type == "bilstm":
        model.add(Bidirectional(LSTM(64, activation="relu", dropout=0.25, kernel_regularizer=regularizer, bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5), return_sequences=True, recurrent_dropout=0.25)))
    
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=0.001))
    
    return model

print("Choose model type: rnn, lstm, or bilstm")
chosen_model = input()

model_y1 = create_model(chosen_model, vocab_size, MAX_LENGTH, len(y1c))
model_y2 = create_model(chosen_model, vocab_size, MAX_LENGTH, len(y2c))

model_y1.fit(tokenized_X_train, y1_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)
model_y2.fit(tokenized_X_train, y2_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

y1_pred = model_y1.predict(tokenized_X_test)
y1_pred = y1_pred.argmax(axis=1)

y2_pred = model_y2.predict(tokenized_X_test)
y2_pred = y2_pred.argmax(axis=1)

def print_metrics(y_true, y_pred, task):
    print(f"\n{task} Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))

print_metrics(y1_test, y1_pred, "POS Tagging")
print_metrics(y2_test, y2_pred, "Named Entity Recognition")
