# TimeVAE for Synthetic Timeseries Data Generation

TimeVAE implementation in keras/tensorflow implementation of timevae:

TimeVAE is used for synthetic time-series data generation. See paper:

https://arxiv.org/abs/2111.08095

# Project Information

The methodology uses the Variational Autoencoder architecture. The decoder architecture is modified to include interpretable components of time-series, namely, level, trend, and seasonality. The repo also contains two other baselines: a dense VAE and a convolutional VAE. The dense VAE is a simple VAE with dense layers in the encoder and decoder. The convolutional VAE is a VAE with convolutional layers in the encoder and decoder. The convolutional VAE is referred to as the base model in the paper. See `./src/vae/` for the implementation of the models. Note that `vae_base.py` script contains an abstract super-class. It doesnt actually represent TimeVAE-Base.

See script `./src/vae_pipeline.py` for usage of the TimeVAE model. The `run_vae_pipeline` function in the script is the main function. Specify the dataset name and the model type near the bottom of the script and run. The pipeline does the following:

- Loads the data
- Splits the data into train and test sets
- Scales the two sets using a custom MinMax scaler
- Instantiates the VAE model of given type (one of `vae_dense`, `vae_conv`, or `timeVAE`).
- Trains the model. Trained model is saved in `./outputs/models` under a specific directory created for the given dataset. The directory name is the same as the dataset name.
- Generates posterior samples for comparison with training samples.
- Generates prior samples for synthetic data generation.
  - Saves the prior samples in `./outputs/gen_data` under a specific directory created for the given dataset. The directory name is the same as the dataset name.
  - Performs t-SNE transformation of the original train data and the prior samples for visualization. Creates a plot of the t-SNE transformed data and saves it in `./outputs/tsne` under a specific directory created for the given dataset. The directory name is the same as the dataset name.

## Usage

- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Prepare your data in the format of numpy array with shape `(n_samples, n_timesteps, n_features)` and save it in the `./data/` folder in the `.npz` format. The file name without the extension will be used as the dataset name. For example, if your file is named `my_data.npz`, the dataset name will be `my_data`.
- Update the name of dataset using the `dataset_name` variable in the `./src/run_vae_pipeline.py` script.
- Specify the model using the `model_name` variable. You can choose between `vae_dense`, `vae_conv`, and `timeVAE` for the dense, convolutional (referred to base model in paper), and interpretable versions of TimeVAE, respectively.
- Review and set the hyperparameters for the model in the `./src/config/hyperparameters.yaml` file. The four key hyperparameters are `latent_dim`, `hidden_layer_sizes`, `reconstruction_wt`, and `batch_size`. More details are as follows:
  - `latent_dim` - the number of latent dimensions. Default value is 8.
  - `hidden_layer_sizes` - the number of hidden units in the encoder and decoder. For the dense model, it refers to the number of hidden units in the dense layers. For the convolutional model, it refers to the number of filters in the convolutional layers. Default value is [50, 100, 200].
  - `reconstruction_wt` - A parameter which decides to the weight to be given to the reconstruction loss. The VAE loss is the sum of two losses: the reconstruction loss and the KL divergence loss. The reconstruction loss is weighted by this parameter. The higher the weight to the reconstruction loss, the more emphasis is given to the reconstruction loss, and lower to the KL divergence loss - which means that the embeddings are less likely to conform to the prior distribution. This will likely erode the quality of prior samples although the posterior samples may be better. Default value is 3.0
  - `batch_size` - the batch size for training. Default value is 16.
- For the `TimeVAE` model, you can also set the hyperparameters specific to the interpretability components. Setting `trend_poly`=0, and `custom_seas`=null, and `use_residual_conn`=true will result in the base model (i.e. without interpretable components).
  - `trend_poly`: To specify the degree of the polynomial trend component, set `trend_poly` to an integer value. For example, setting `trend_poly` to 2 will include a quadratic trend component.
  - `custom_seas`: To specify custom seasonalities, set `custom_seas` to be a list of tuples, where each tuple contains the number of seasons and the length of each season for that frequency. For example, [(7, 1)] indicates seasonality with 7 seasons, each of length 1. You can specify multiple such seasonalities. For example, [(7, 1), (12, 1)] indicates 2 seasonalities: first with 7 seasons of length 1 and secodn with 12 seasons of length 1. Default value is null which represents no custom seasonalities.
  - `use_residual_conn`: This specifies if you want to have a residual connection. To only use the interpretable components, and not the residual connection, set `use_residual_conn` to false. However, for best performance, it is recommended to use the residual connection. Default value is true.
- Note that the default hyperparameters in the `hyperparameters.yaml` file tend to perform well on most datasets. Still, you may want to tune the hyperparameters for your specific dataset.
- Run the script using `python src/vae_pipeline.py`. Three types of outputs will be saved in the `./outputs/` folder: generated samples, trained model artifacts, and t-sne charts.
  - The trained model artifacts will be saved in the `./outputs/models/<dataset_name>/` folder.
  - Generated prior synthetic samples will be saved in the `./outputs/gen_data/<dataset_name>/` folder. Samples are saved in the `.npz` format.
  - Scatter plots of original and generated samples after t-SNE transformation are saved in the `./outputs/tsne/<dataset_name>/` folder.

## Requirements

Dependencies are listed in the file `requirements.txt`.
You can install these packages by running the following command from the root of your project directory:

```python
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## Contact Information

You can contact the lead-author of the paper at: <lead author first name>.<lead author last name>@gmail.com

See the author names in the paper linked above.
