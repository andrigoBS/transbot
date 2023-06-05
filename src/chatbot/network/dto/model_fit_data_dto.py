class ModelFitDataDTO:
    def __init__(self, encoder_input_data, decoder_input_data, decoder_output_data):
        self.encoder_input_data = encoder_input_data
        self.decoder_input_data = decoder_input_data
        self.decoder_output_data = decoder_output_data
