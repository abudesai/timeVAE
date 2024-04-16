# TimeVAE for Synthetic Timeseries Data Generation

TimeVAE implementation in keras/tensorflow implementation of timevae:

TimeVAE is used for synthetic time-series data generation. See paper:

https://arxiv.org/abs/2111.08095

# Project Information

The methodology uses the Variational Autoencoder architecture. The decoder architecture is modified to include interpretable components of time-series, namely, level, trend, and seasonality.

`vae_conv_I_model.py` script contains the interpretable version of TimeVAE. See class `VariationalAutoencoderConvInterpretable`.

`vae_conv_model.py` contains the base version of TimeVAE. See class `VariationalAutoencoderConv`

The VariationalAutoencoderConvInterpretable in `vae_conv_I_model.py` can also be used as base version by disabling the interpretability-related arguments during class initialization.

See script `test_vae.py` for usage of the TimeVAE model.

Note that `vae_base.py` script contains an abstract super-class. It doesnt actually represent TimeVAE-Base.

## Usage

- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Prepare your data in the format of numpy array with shape `(n_samples, n_timesteps, n_features)` and save it in the `./datasets/` folder in the `.npz` format.
- Update the path to your data in the `test_vae.py` script. You can specify the file name in the `input_file` variable.
- Choose the model near the top of the `test_vae.py` script using the `vae_type` variable. You can choose between `vae_dense`, `vae_conv`, and `timeVAE` for the dense, convolutional (referred to base model in paper), and interpretable versions of TimeVAE, respectively.
- Set the hyperparameters for the model. Key hyperparameters are:
  - `latent_dim` - the number of latent dimensions. Default value is 8.
  - `hidden_layer_sizes` - the number of hidden units in the encoder and decoder. For the dense model, it refers to the number of hidden units in the dense layers. For the convolutional model, it refers to the number of filters in the convolutional layers. Default value is [50, 100, 200]
  - `reconstruction_wt` - A parameter which decides to the weight to be given to the reconstruction loss. The VAE loss is the sum of two losses: the reconstruction loss and the KL divergence loss. The reconstruction loss is weighted by this parameter. The higher the weight to the reconstruction loss, the more emphasis is given to the reconstruction loss, and lower to the KL divergence loss - which means that the embeddings are less likely to conform to the prior distribution. This will likely erode the quality of prior samples although the posterior samples may be better. Default value is 3.0
  - `batch_size` - the batch size for training the model. Defaut value is 32.
- You can also set the hyperparameters specific to the interpretability components of the TimeVAE model.
- Note that the default hyperparameters tend to perform well on most datasets. Still, you may need to tune the hyperparameters for your specific dataset.
- Run the script using `python test_vae.py`. Note that the script also preprocesses the data using a custom MinMax scaler.
  - The trained model will be saved in the `./models/` folder.
  - Generated prior synthetic samples will be saved in the `./outputs/` folder.

## Requirements

Dependencies are listed in the file `requirements.txt`.
You can install these packages by running the following command from the root of your project directory:

```python
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## Contact Information

You can contact the lead-author of the paper at: <lead_author_first_name>.<lead_author_last_name>@gmail.com

See the author names in the paper linked above.
