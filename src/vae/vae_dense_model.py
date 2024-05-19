import joblib
import os, warnings, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from vae.vae_base import BaseVariationalAutoencoder, Sampling


class VariationalAutoencoderDense(BaseVariationalAutoencoder):
    model_name = "VAE_Dense"

    def __init__(self, hidden_layer_sizes, **kwargs):
        super(VariationalAutoencoderDense, self).__init__(**kwargs)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.hidden_layer_sizes = hidden_layer_sizes

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.compile(optimizer=Adam())

    def _get_encoder(self):
        self.encoder_inputs = Input(
            shape=(self.seq_len, self.feat_dim), name="encoder_input"
        )

        x = Flatten()(self.encoder_inputs)
        for i, M_out in enumerate(self.hidden_layer_sizes):
            x = Dense(M_out, activation="relu", name=f"enc_dense_{i}")(x)

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        encoder_output = Sampling()([z_mean, z_log_var])
        self.encoder_output = encoder_output

        encoder = Model(
            self.encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder"
        )
        # encoder.summary()
        return encoder

    def _get_decoder(self):
        decoder_inputs = Input(shape=(self.latent_dim,), name="decoder_input")

        x = decoder_inputs
        for i, M_out in enumerate(reversed(self.hidden_layer_sizes)):
            x = Dense(M_out, activation="relu", name=f"dec_dense_{i}")(x)
        x = Dense(self.seq_len * self.feat_dim, name="decoder_final_dense")(x)
        self.decoder_outputs = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        decoder = Model(decoder_inputs, self.decoder_outputs, name="decoder")
        # decoder.summary()
        return decoder

    @classmethod
    def load(cls, model_dir: str) -> "VariationalAutoencoderDense":
        params_file = os.path.join(model_dir, f"{cls.model_name}_parameters.pkl")
        dict_params = joblib.load(params_file)
        vae_model = VariationalAutoencoderDense(**dict_params)
        vae_model.load_weights(model_dir)
        vae_model.compile(optimizer=Adam())
        return vae_model
