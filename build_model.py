from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, LSTM, Embedding, BatchNormalization
from tensorflow.keras.layers import Dropout, Concatenate


def lstm_model(max_length, vocab_size):
    # CNN 입력 처리
    cnn_input = Input(shape=(1280,), name="cnn_input")
    cnn_input = Dropout(0.8)(cnn_input)
    cnn_layer1 = Dense(256, activation='relu')(cnn_input)
    cnn_layer2 = BatchNormalization()(cnn_layer1)

    # Text 입력 처리
    text_input = Input(shape=(max_length,), name="text_input")
    embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(text_input)
    embedding = Dropout(0.8)(embedding)
    lstm_output = LSTM(units=256)(embedding)

    # CNN + Text 결합
    # add_layers = Add()([cnn_layer2, lstm_output])
    concat_layers = Concatenate(axis=-1)([cnn_layer2, lstm_output])
    decoder1 = Dense(256, activation='relu')(concat_layers)
    output = Dense(vocab_size, activation='softmax')(decoder1)

    # 모델 정의
    model = Model(inputs=[cnn_input, text_input], outputs=output)
    return model