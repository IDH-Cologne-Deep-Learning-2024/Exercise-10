# IMPORTANT
#  This has worked before, as of now i cannot reproduce/confirm it due to massive problems concerning the compatability of tensorflow with my current pythion version
# I have tried virtual enviroments, using pyenv to install a previous version of python and still not found succes


import pandas 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, TimeDistributed, SimpleRNN
from sklearn.metrics import classification_report

# Load the dataset
file_path = 'gmb.csv' 
data = pd.read_csv(file_path, names=["Word", "POS", "NE"], skip_blank_lines=True)

data = data.dropna().reset_index(drop=True)

# Preprocessing
words = list(set(data["Word"].values))
words.append("ENDPAD")
n_words = len(words)

pos_tags = list(set(data["POS"].values))
n_pos = len(pos_tags)

ne_tags = list(set(data["NE"].values))
n_ne = len(ne_tags)

# Map
word2idx = {w: i + 1 for i, w in enumerate(words)}
pos2idx = {t: i for i, t in enumerate(pos_tags)}
ne2idx = {t: i for i, t in enumerate(ne_tags)}

# Encode
data["Word_idx"] = data["Word"].map(word2idx)
data["POS_idx"] = data["POS"].map(pos2idx)
data["NE_idx"] = data["NE"].map(ne2idx)

agg_func = lambda x: list(x)
sentences = data.groupby("Word").apply(agg_func)

X = [[word2idx[w] for w in s["Word"]] for _, s in sentences]
y_pos = [[pos2idx[t] for t in s["POS"]] for _, s in sentences]
y_ne = [[ne2idx[t] for t in s["NE"]] for _, s in sentences]

# Padding
max_len = max([len(seq) for seq in X])
X = pad_sequences(X, maxlen=max_len, padding="post")
y_pos = pad_sequences(y_pos, maxlen=max_len, padding="post")
y_ne = pad_sequences(y_ne, maxlen=max_len, padding="post")

# 1hot
y_pos = [to_categorical(i, num_classes=n_pos) for i in y_pos]
y_ne = [to_categorical(i, num_classes=n_ne) for i in y_ne]

# split
X_train, X_test, y_pos_train, y_pos_test, y_ne_train, y_ne_test = train_test_split(
    X, y_pos, y_ne, test_size=0.1, random_state=42
)

# function 
def build_model(model_type="LSTM"):
    model = Sequential()
    model.add(Embedding(input_dim=n_words + 1, output_dim=50, input_length=max_len))
    if model_type == "LSTM":
        model.add(LSTM(units=100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    elif model_type == "BiLSTM":
        model.add(Bidirectional(LSTM(units=100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    elif model_type == "RNN":
        model.add(SimpleRNN(units=100, return_sequences=True, dropout=0.2))
    model.add(TimeDistributed(Dense(n_pos + n_ne, activation="softmax")))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# train/eval
for model_name in ["LSTM", "BiLSTM", "RNN"]:
    print(f"Training {model_name}...")
    model = build_model(model_type=model_name)
    model.fit(
        X_train,
        [np.array(y_pos_train), np.array(y_ne_train)],
        validation_split=0.1,
        batch_size=32,
        epochs=5,
        verbose=1
    )
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_pred_pos = np.argmax(y_pred[0], axis=-1)
    y_pred_ne = np.argmax(y_pred[1], axis=-1)

    y_true_pos = np.argmax(np.array(y_pos_test), axis=-1)
    y_true_ne = np.argmax(np.array(y_ne_test), axis=-1)

    print(f"{model_name} POS classification report:")
    print(classification_report(y_true_pos.flatten(), y_pred_pos.flatten()))

    print(f"{model_name} NE classification report:")
    print(classification_report(y_true_ne.flatten(), y_pred_ne.flatten()))