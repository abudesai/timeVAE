import os, warnings, sys
from re import T

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv1D,
    Flatten,
    Dense,
    Conv1DTranspose,
    Reshape,
    Input,
    Layer,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from vae.vae_base import BaseVariationalAutoencoder, Sampling


class TrendLayer(Layer):
    def __init__(self, feat_dim, trend_poly, seq_len, **kwargs):
        super(TrendLayer, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.trend_poly = trend_poly
        self.seq_len = seq_len
        self.trend_dense1 = Dense(
            self.feat_dim * self.trend_poly, activation="relu", name="trend_params"
        )
        self.trend_dense2 = Dense(self.feat_dim * self.trend_poly, name="trend_params2")
        self.reshape_layer = Reshape(target_shape=(self.feat_dim, self.trend_poly))

    def call(self, z):
        trend_params = self.trend_dense1(z)
        trend_params = self.trend_dense2(trend_params)
        trend_params = self.reshape_layer(trend_params)  # shape: N x D x P

        lin_space = (
            tf.range(0, float(self.seq_len), 1) / self.seq_len
        )  # shape of lin_space: 1d tensor of length T
        poly_space = tf.stack(
            [lin_space ** float(p + 1) for p in range(self.trend_poly)], axis=0
        )  # shape: P x T

        trend_vals = tf.matmul(trend_params, poly_space)  # shape (N, D, T)
        trend_vals = tf.transpose(trend_vals, perm=[0, 2, 1])  # shape: (N, T, D)
        trend_vals = tf.cast(trend_vals, tf.float32)

        return trend_vals


class SeasonalLayer(Layer):
    def __init__(self, feat_dim, seq_len, custom_seas, **kwargs):
        super(SeasonalLayer, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.custom_seas = custom_seas
        self.dense_layers = [
            Dense(feat_dim * num_seasons, name=f"season_params_{i}")
            for i, (num_seasons, len_per_season) in enumerate(custom_seas)
        ]
        self.reshape_layers = [
            Reshape(target_shape=(feat_dim, num_seasons))
            for num_seasons, len_per_season in custom_seas
        ]

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = tf.range(num_seasons)[:, None] + tf.zeros(
            (num_seasons, len_per_season), dtype=tf.int32
        )
        season_indexes = tf.reshape(season_indexes, [-1])
        # Ensure the length matches seq_len
        season_indexes = tf.tile(season_indexes, [self.seq_len // len_per_season + 1])[
            : self.seq_len
        ]
        return season_indexes

    def call(self, z):
        N = tf.shape(z)[0]
        ones_tensor = tf.ones(shape=[N, self.feat_dim, self.seq_len], dtype=tf.int32)

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)  # shape: (N, D * S)
            season_params = self.reshape_layers[i](season_params)  # shape: (N, D, S)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            )  # shape: (T, )

            dim2_idxes = ones_tensor * tf.reshape(
                season_indexes_over_time, shape=(1, 1, -1)
            )  # shape: (N, D, T)
            season_vals = tf.gather(
                season_params, dim2_idxes, batch_dims=-1
            )  # shape (N, D, T)

            all_seas_vals.append(season_vals)

        all_seas_vals = K.stack(all_seas_vals, axis=-1)  # shape: (N, D, T, S)
        all_seas_vals = tf.reduce_sum(all_seas_vals, axis=-1)  # shape (N, D, T)
        all_seas_vals = tf.transpose(all_seas_vals, perm=[0, 2, 1])  # shape (N, T, D)
        return all_seas_vals

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len, self.feat_dim)


class TimeVAE(BaseVariationalAutoencoder):
    model_name = "TimeVAE"

    def __init__(
        self,
        hidden_layer_sizes,
        trend_poly=0,
        custom_seas=None,
        use_residual_conn=True,
        **kwargs,
    ):
        """
        hidden_layer_sizes: list of number of filters in convolutional layers in encoder and residual connection of decoder.
        trend_poly: integer for number of orders for trend component. e.g. setting trend_poly = 2 will include linear and quadratic term.
        custom_seas: list of tuples of (num_seasons, len_per_season).
            num_seasons: number of seasons per cycle.
            len_per_season: number of epochs (time-steps) per season.
        use_residual_conn: boolean value indicating whether to use a residual connection for reconstruction in addition to
        trend, generic and custom seasonalities.
        """

        super(TimeVAE, self).__init__(**kwargs)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.compile(optimizer=Adam())

    def _get_encoder(self):
        encoder_inputs = Input(
            shape=(self.seq_len, self.feat_dim), name="encoder_input"
        )
        x = encoder_inputs
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = Conv1D(
                filters=num_filters,
                kernel_size=3,
                strides=2,
                activation="relu",
                padding="same",
                name=f"enc_conv_{i}",
            )(x)

        x = Flatten(name="enc_flatten")(x)

        # save the dimensionality of this last dense layer before the hidden state layer.
        # We need it in the decoder.
        self.encoder_last_dense_dim = x.shape[-1]

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

        encoder_output = Sampling()([z_mean, z_log_var])
        self.encoder_output = encoder_output

        encoder = Model(
            encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder"
        )
        # encoder.summary()
        return encoder

    def _get_decoder(self):
        decoder_inputs = Input(shape=(self.latent_dim,), name="decoder_input")

        outputs = None
        outputs = self.level_model(decoder_inputs)
        # trend polynomials
        if self.trend_poly is not None and self.trend_poly > 0:
            trend_vals = TrendLayer(self.feat_dim, self.trend_poly, self.seq_len)(
                decoder_inputs
            )
            outputs = trend_vals if outputs is None else outputs + trend_vals

        # custom seasons
        if self.custom_seas is not None and len(self.custom_seas) > 0:
            # cust_seas_vals = self.custom_seasonal_model(decoder_inputs)
            cust_seas_vals = SeasonalLayer(
                feat_dim=self.feat_dim,
                seq_len=self.seq_len,
                custom_seas=self.custom_seas,
            )(decoder_inputs)
            outputs = cust_seas_vals if outputs is None else outputs + cust_seas_vals

        if self.use_residual_conn:
            residuals = self._get_decoder_residual(decoder_inputs)
            outputs = residuals if outputs is None else outputs + residuals

        if outputs is None:
            raise ValueError(
                "Error: No decoder model to use. "
                "You must use one or more of:"
                "trend, generic seasonality(ies), custom seasonality(ies), "
                "and/or residual connection. "
            )

        decoder = Model(decoder_inputs, [outputs], name="decoder")
        return decoder

    def level_model(self, z):
        level_params = Dense(self.feat_dim, name="level_params", activation="relu")(z)
        level_params = Dense(self.feat_dim, name="level_params2")(level_params)
        level_params = Reshape(target_shape=(1, self.feat_dim))(
            level_params
        )  # shape: (N, 1, D)

        ones_tensor = tf.ones(
            shape=[1, self.seq_len, 1], dtype=tf.float32
        )  # shape: (1, T, D)

        level_vals = level_params * ones_tensor
        # print('level_vals', tf.shape(level_vals))
        return level_vals

    def _get_decoder_residual(self, x):
        x = Dense(self.encoder_last_dense_dim, name="dec_dense", activation="relu")(x)
        x = Reshape(target_shape=(-1, self.hidden_layer_sizes[-1]), name="dec_reshape")(
            x
        )

        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = Conv1DTranspose(
                filters=num_filters,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu",
                name=f"dec_deconv_{i}",
            )(x)

        # last de-convolution
        x = Conv1DTranspose(
            filters=self.feat_dim,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            name=f"dec_deconv__{i+1}",
        )(x)

        x = Flatten(name="dec_flatten")(x)
        x = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")(x)
        residuals = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        return residuals

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        super().save(model_dir)  # Save common parameters and weights

        # self.custom_seas is a Keras TrackedList, need to convert it
        # back to list of tuples so it is serializable
        if self.custom_seas is not None:
            self.custom_seas = [
                (int(num_seasons), int(len_per_season))
                for num_seasons, len_per_season in self.custom_seas
            ]

        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(
                self.hidden_layer_sizes
            ),  # make sure it's a list which is serializable
            "trend_poly": self.trend_poly,
            "custom_seas": self.custom_seas,
            "use_residual_conn": self.use_residual_conn,
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

    @classmethod
    def load(cls, model_dir: str) -> "TimeVAE":
        params_file = os.path.join(model_dir, f"{cls.model_name}_parameters.pkl")
        dict_params = joblib.load(params_file)
        vae_model = TimeVAE(**dict_params)
        vae_model.load_weights(model_dir)
        vae_model.compile(optimizer=Adam())
        return vae_model
