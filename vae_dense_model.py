
import os, warnings, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils import get_mnist_data, draw_orig_and_post_pred_sample, plot_latent_space
from vae_base import BaseVariationalAutoencoder, Sampling


class VariationalAutoencoderDense(BaseVariationalAutoencoder):  

    def __init__(self,  hidden_layer_sizes,  **kwargs  ):
        super(VariationalAutoencoderDense, self).__init__(**kwargs)

        self.hidden_layer_sizes = hidden_layer_sizes  

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()        


    def _get_encoder(self):
        self.encoder_inputs = Input(shape=(self.seq_len, self.feat_dim), name='encoder_input')

        x = Flatten()(self.encoder_inputs)
        for i, M_out in enumerate(self.hidden_layer_sizes):
            x = Dense(M_out, activation='relu', name=f'enc_dense_{i}')(x)

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        encoder_output = Sampling()([z_mean, z_log_var])     
        self.encoder_output = encoder_output        
        
        encoder = Model(self.encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder")
        # encoder.summary()
        return encoder 


    def _get_decoder(self):
        decoder_inputs = Input(shape=(self.latent_dim), name='decoder_input')

        x = decoder_inputs
        for i, M_out in enumerate(reversed(self.hidden_layer_sizes)):
            x = Dense(M_out, activation='relu', name=f'dec_dense_{i}')(x)
        x = Dense(self.seq_len * self.feat_dim, name='decoder_final_dense')(x)
        self.decoder_outputs = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        decoder = Model(decoder_inputs, self.decoder_outputs, name="decoder")
        # decoder.summary()
        return decoder



#####################################################################################################
#####################################################################################################


if __name__ == '__main__':    
    
    mnist_digits = get_mnist_data()
    print('data shape:', mnist_digits.shape)
    N, T, D = mnist_digits.shape

    vae = VariationalAutoencoderDense(
        seq_len=T,
        feat_dim = D,
        latent_dim = 2,
        hidden_layer_sizes=[200,100],
    )

    vae.compile(optimizer=Adam())

    # vae.summary()



    r = vae.fit(mnist_digits, epochs=10, batch_size=128, shuffle=True)


    x_decoded = vae.predict(mnist_digits)
    print('x_decoded.shape', x_decoded.shape)


    # compare original and posterior predictive (reconstructed) samples
    draw_orig_and_post_pred_sample(mnist_digits, x_decoded, n=5)


    # generate prior predictive samples by sampling from latent space
    plot_latent_space(vae, 30, figsize=15)