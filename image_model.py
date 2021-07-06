from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model

def generator(photo, caption, MAX_LEN):
    n_samples = 0
    X = []
    y_in = []
    y_out = []
    for k, vv in caption.items():
        for v in vv:
            for i in range(1, len(v)):
                X.append(photo[k])

                in_seq= [v[:i]]
                out_seq = v[i]

                in_seq = pad_sequences(in_seq, maxlen=MAX_LEN, padding='post', truncating='post')[0]
                out_seq = to_categorical([out_seq], num_classes=VOCAB_SIZE)[0]

                y_in.append(in_seq)
                y_out.append(out_seq)
            
    return np.array(X), np.array(y_in, dtype="float64"), np.array(y_out, dtype="float64")

def return_model(embedding_size, max_len, vocab_size):
    image_model = Sequential()

    image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
    image_model.add(RepeatVector(max_len))

    image_model.summary()

    language_model = Sequential()

    language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
    language_model.add(LSTM(256, return_sequences=True))
    language_model.add(TimeDistributed(Dense(embedding_size)))

    language_model.summary()

    conca = Concatenate()([image_model.output, language_model.output])
    x = LSTM(128, return_sequences=True)(conca)
    x = LSTM(512, return_sequences=False)(x)
    x = Dense(vocab_size)(x)
    out = Activation('softmax')(x)
    model = Model(inputs=[image_model.input, language_model.input], outputs = out)

    # model.load_weights("../input/model_weights.h5")
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model