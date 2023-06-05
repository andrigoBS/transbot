import keras
from src.chatbot.network.dto.model_fit_data_dto import ModelFitDataDTO
from src.chatbot.network.dto.model_params_dto import ModelParamsDTO
from src.file_path_helper import FilePathHelper


class Sec2SecModel:
    def __init__(self):
        self.model_path_name = FilePathHelper().get_model_file_path()
        self.model = None

    def load(self):
        self.model = keras.models.load_model(self.model_path_name)

    def predict_phrase(self, phrase_sequence, target_seq):
        return self.model.predict([phrase_sequence, target_seq])

    def create(self, max_phrase_size, vocab_size, params: ModelParamsDTO):
        encoder_inputs = keras.layers.Input(shape=(max_phrase_size,), batch_size=params.batch_size)
        encoder_outputs, encoder_states = self.__create_encoder(encoder_inputs, vocab_size, max_phrase_size)

        decoder_inputs = keras.layers.Input(shape=(max_phrase_size,), batch_size=params.batch_size)
        decoder_outputs = self.__create_decoder(
            decoder_inputs, encoder_outputs, encoder_states, vocab_size, max_phrase_size
        )

        dense_outputs = self.__create_exit_layer(decoder_outputs, vocab_size)
        self.model = keras.Model([encoder_inputs, decoder_inputs], dense_outputs)

        summary = []
        self.model.summary(print_fn=lambda line: summary.append(line))
        return summary

    def fit(self, model_fit_data_dto: ModelFitDataDTO, params: ModelParamsDTO):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=params.metrics)
        history = self.model.fit(
            [model_fit_data_dto.encoder_input_data, model_fit_data_dto.decoder_input_data],
            model_fit_data_dto.decoder_output_data,
            epochs=params.epochs,
            batch_size=params.batch_size,
            validation_split=params.validation_split,
            shuffle=params.shuffle
        )

        self.model.save(self.model_path_name)
        return history

    def plot_model(self, file_name):
        keras.utils.plot_model(self.model, show_shapes=True, to_file=file_name)

    def __create_encoder(self, encoder_inputs, vocab_size, max_phrase_size):
        encoder_embedding = keras.layers.Embedding(vocab_size, max_phrase_size, mask_zero=True)(encoder_inputs)
        encoder_lstm = keras.layers.LSTM(max_phrase_size, return_state=True)
        bidirectional_lstm = keras.layers.Bidirectional(encoder_lstm)

        encoder_outputs, forward_h, forward_c, backward_h, backward_c = bidirectional_lstm(encoder_embedding)

        encoder_outputs = keras.layers.Dense(max_phrase_size, activation='sigmoid')(encoder_outputs)

        state_h = keras.layers.Concatenate()([forward_h, backward_h])
        state_c = keras.layers.Concatenate()([forward_c, backward_c])

        encoder_states = [state_h, state_c]

        return encoder_outputs, encoder_states

    def __create_decoder(self, decoder_inputs, encoder_outputs, encoder_states, vocab_size, max_phrase_size):
        concatened = keras.layers.AdditiveAttention()([decoder_inputs, encoder_outputs])
        decoder_embedding = keras.layers.Embedding(vocab_size, max_phrase_size, mask_zero=True)(concatened)

        decoder_lstm = keras.layers.LSTM(max_phrase_size * 2, return_state=True, return_sequences=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

        return decoder_outputs

    def __create_exit_layer(self, decoder_outputs, vocab_size):
        dense_layer = keras.layers.Dense(vocab_size, activation='softmax')
        dense_outputs = keras.layers.TimeDistributed(dense_layer)(decoder_outputs)

        return dense_outputs
