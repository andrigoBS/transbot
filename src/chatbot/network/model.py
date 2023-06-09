import random
import keras.layers
import keras.utils

from src.file_path_helper import FilePathHelper


class Sec2SecModel:
    def __init__(self):
        self.model_path_name = FilePathHelper().get_model_file_path()
        self.model = None

    def load(self):
        self.model = keras.models.load_model(self.model_path_name)

    def predict_phrase(self, phrase_sequence, target_seq):
        return self.model.predict([phrase_sequence, target_seq])

    def create(self, max_phrase_size, vocab_size):
        encoder_inputs = keras.layers.Input(shape=(max_phrase_size, vocab_size), batch_size=1)
        decoder_inputs = keras.layers.Input(shape=(max_phrase_size, vocab_size), batch_size=1)

        encoder_outputs, encoder_states = self.__create_encoder(encoder_inputs, max_phrase_size)
        decoder_outputs = self.__create_decoder(decoder_inputs, encoder_states, max_phrase_size)
        attention = self.__create_attention(encoder_outputs, decoder_outputs, max_phrase_size)
        dense_outputs = self.__create_exit_layer(attention, vocab_size)

        self.model = keras.Model([encoder_inputs, decoder_inputs], dense_outputs)
        summary = []
        self.model.summary(print_fn=lambda line: summary.append(line))
        return summary

    def fit(self, get_conversation_data, conversations_size, epochs, metrics):
        def data_generator():
            conversations = []
            for i in range(conversations_size):
                conversations.append(get_conversation_data(i))

            while True:
                random.shuffle(conversations)
                for conversation in conversations:
                    encoder_input_data, decoder_input_data, decoder_output_data = conversation
                    for i in range(len(encoder_input_data)):
                        yield [encoder_input_data[i:i+1], decoder_input_data[i:i+1]], decoder_output_data[i:i+1]

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=metrics)
        history = self.model.fit_generator(
            data_generator(),
            epochs=epochs,
            steps_per_epoch=conversations_size,
            shuffle=False
        )

        self.model.save(self.model_path_name)
        return history

    def plot_model(self, file_name):
        keras.utils.plot_model(self.model, show_shapes=True, to_file=file_name)

    def __create_encoder(self, encoder_inputs, max_phrase_size):
        encoder_lstm = keras.layers.LSTM(
            max_phrase_size, return_state=True, return_sequences=True, recurrent_initializer='glorot_uniform'
        )
        bidirectional_lstm = keras.layers.Bidirectional(encoder_lstm, merge_mode='sum')

        encoder_outputs, forward_h, forward_c, backward_h, backward_c = bidirectional_lstm(encoder_inputs)

        state_h = keras.layers.Concatenate()([forward_h, backward_h])
        state_c = keras.layers.Concatenate()([forward_c, backward_c])
        state_h = keras.layers.Dense(max_phrase_size, activation='relu')(state_h)
        state_c = keras.layers.Dense(max_phrase_size, activation='relu')(state_c)

        encoder_states = [state_h, state_c]

        return encoder_outputs, encoder_states

    def __create_attention(self, query, value, max_phrase_size):
        attention = keras.layers.MultiHeadAttention(key_dim=max_phrase_size, num_heads=int(max_phrase_size/10))

        output = attention(value=value, query=query)
        output = keras.layers.Add()([value, output])
        output = keras.layers.LayerNormalization()(output)

        return output

    def __create_decoder(self, decoder_inputs, encoder_states, max_phrase_size):
        decoder_lstm = keras.layers.LSTM(
            max_phrase_size, return_sequences=True, recurrent_initializer='glorot_uniform'
        )
        decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        return decoder_outputs

    def __create_exit_layer(self, decoder_outputs, vocab_size):
        dense_layer = keras.layers.Dense(vocab_size, activation='softmax')
        dense_outputs = keras.layers.TimeDistributed(dense_layer)(decoder_outputs)
        return dense_outputs
