# TimeVAE for Synthetic Timeseries Data Generation

TimeVAE is a model designed for generating synthetic time-series data using a Variational Autoencoder (VAE) architecture with interpretable components like level, trend, and seasonality. This repository includes the implementation of TimeVAE, as well as two baseline models: a dense VAE and a convolutional VAE.

## Paper Reference

For a detailed explanation of the methodology, see the paper: [TIMEVAE: A VARIATIONAL AUTO-ENCODER FOR
MULTIVARIATE TIME SERIES GENERATION](https://arxiv.org/abs/2111.08095).

## Project Information

This project implements the Variational Autoencoder architecture with modifications to the decoder to include interpretable components of time-series data: level, trend, and seasonality. Additionally, it provides two other baseline models:

- **Dense VAE**: A simple VAE with dense layers in the encoder and decoder.
- **Convolutional VAE**: A VAE with convolutional layers in the encoder and decoder, referred to as the base model in the paper.

See `./src/vae/` for the implementation of these models. Note that `vae_base.py` contains an abstract superclass and does not represent TimeVAE-Base.

## Project Structure

```plaintext
TimeVAE/
├── data/                         # Folder for datasets
├── outputs/                      # Folder for model outputs
│   ├── gen_data/                 # Folder for generated samples
│   ├── models/                   # Folder for model artifacts
│   └── tsne/                     # Folder for t-SNE plots
├── src/                          # Source code
│   ├── config/                   # Configuration files
│   │   └── hyperparameters.yaml  # Hyperparameters settings
│   ├── vae/                      # VAE models implementation
│   │   ├── timevae.py            # Main TimeVAE model
│   │   ├── vae_base.py           # Abstract superclass
│   │   ├── vae_conv_model.py     # Convolutional VAE model (base model)
│   │   ├── vae_dense_model.py    # Dense VAE model
│   │   └── vae_utils.py          # utils to create, train, and use VAE models
│   ├── data_utils.py             # utils for data loading, splitting and scaling
│   ├── paths.py                  # path variables for config file, data, models, and outputs
│   ├── vae_pipeline.py           # Main pipeline script
│   └── visualize.py              # Scripts for visualization, including t-SNE plots
├── LICENSE.md                    # License information
├── README.md                     # Readme file
└── requirements.txt              # Dependencies

```

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage

1. **Prepare Data**: Save your data as a numpy array with shape `(n_samples, n_timesteps, n_features)` in the `./data/` folder in `.npz` format. The filename without the extension will be used as the dataset name (e.g., `my_data.npz` will be referred to as `my_data`). Alternatively, use one of the existing datasets provided in the `./data/` folder.

2. **Configure Pipeline**:

   - Update the dataset name and model type in `./src/vae_pipeline.py`:
     ```python
     dataset = "my_data"  # Your dataset name
     model_name = "timeVAE"  # Choose between vae_dense, vae_conv, or timeVAE
     ```
   - Set hyperparameters in `./src/config/hyperparameters.yaml`. Key hyperparameters include `latent_dim`, `hidden_layer_sizes`, `reconstruction_wt`, and `batch_size`.

3. **Run the Script**:

   ```bash
   python src/vae_pipeline.py
   ```

4. **Outputs**:
   - Trained models are saved in `./outputs/models/<dataset_name>/`.
   - Generated synthetic data is saved in `./outputs/gen_data/<dataset_name>/` in `.npz` format.
   - t-SNE plots are saved in `./outputs/tsne/<dataset_name>/` in `.png` format.

## Hyperparameters

The four key hyperparameters for the VAE models are:

- `latent_dim`: Number of latent dimensions (default: 8).
- `hidden_layer_sizes`: Number of hidden units or filters (default: [50, 100, 200]).
- `reconstruction_wt`: Weight for the reconstruction loss (default: 3.0).
- `batch_size`: Training batch size (default: 16).

For `timeVAE`:

- `trend_poly`: Degree of polynomial trend component (default: 0).
- `custom_seas`: Custom seasonalities as a list of tuples (default: null).
- `use_residual_conn`: Use residual connection (default: true).

> The default settings for the timeVAE model set it to operate as the base model without interpretable components.

**Note**  
The default hyperparameters in the `./src/config/hyperparameters.yaml` file have been identified after extensive testing on numerous datasets and tend to perform well on most datasets. However, you may want to tune these hyperparameters for your specific dataset.

## FAQs

### Data Structure and Format

#### How do I format my time series data for TimeVAE?

The model expects data to be a 3D numpy array of shape (N, T, D). The load function is looking for a .npz file, but you can replace that logic with your own data loading mechanism.

#### What if I have a single series of length T and dimension D? How do I convert to 3-dimensional data?

Generate windows from your series of length T' where T' < T. Each window becomes a sample in your dataset, creating the N dimension. For example, if you have a time series of length 1000 and you create windows of length 100, you could generate up to 901 windows (with overlap).

#### What's the N in the 3D arrays needed to train the model?

N represents the number of samples or windows generated from your time series. If you have multiple series, you can generate windows from each series and combine them.

#### What if I have multiple series?

You can generate windows from each series and combine them into a single dataset. Make sure each series has the same dimensionality D.

#### What if I have univariate series?

That's fine - the data would still need to be reshaped to be (N, T, D) where D = 1. Simply add an extra dimension to your data.

#### What if I have multivariate series?

TimeVAE naturally handles multivariate time series. Your data shape will be (N, T, D) where D is the number of variables or features in your multivariate series.

#### How do I choose the window length?

This depends on your specific use case - what sized samples are you looking to generate. Two key factors to consider:

1. **Quality of generated samples**: In general, more samples means better quality of generated data, but also longer samples (but not unnecessarily lengthy) improve quality. Since these are counteracting, you need to pick the window length that gives the best balance.
2. **Application requirements**: Consider what length of synthetic time series you need for your downstream applications.

For example, if your data is at daily frequency and has weekly patterns (day-of-week effects), your window length should be at least 7 time steps to capture one full cycle. However, for best results, you might want to use 3-4 weeks (21-28 time steps) to ensure the model properly learns the weekly patterns. Similarly, if you need to generate monthly forecasts, you would want longer windows to ensure the model learns longer-term dependencies.

#### Can TimeVAE handle missing values?

The current implementation doesn't explicitly handle missing values. You would need to preprocess your data to fill in missing values before using TimeVAE.

#### What's the minimum amount of data needed?

This depends on the complexity of patterns in your data and your tolerance for exactly matching patterns in the underlying data. Generally, more complex seasonal patterns or trends require more data. As a rule of thumb, aim for at least a few hundred windows.

### Model Parameters and Configuration

#### How do I choose custom_seas parameters?

Currently, we haven't yet provided an automated solution for this. The original intent for authors was to inject domain knowledge into the model - in this case, users would specify parameters based on known seasonality in their data.

For example, if your data has hourly frequency, you might use `[7, 24]` to represent 7 days with 24 hours each. If your data has daily frequency, you might use `[7, 1]` to represent weekday patterns and `[12,30]` for annual seasonality patterns.

#### How do I choose trend_poly?

The `trend_poly` parameter controls the order of polynomial used to model trends. Set it to 0 for no trend modeling, 1 for linear trends, 2 for quadratic trends, etc. Higher values capture more complex trend patterns but may lead to overfitting.

### Evaluation and Usage

#### How do I evaluate if my synthetic data is good?

The paper presents two primary methods:

1. **Visual inspection**: Visualize t-SNE of original and synthetic data to check if they look similar. This is admittedly subjective but good for sanity checks.
2. **Downstream task performance**: Build models using original vs. synthetic data, and check performance on held-out original data. For example, you could train forecasting models on both datasets and compare their prediction accuracy.

#### How do I evaluate quality of latent space?

This is tricky. In the end, the quality of latent space is best judged by evaluating the quality of generated synthetic samples. See the response to the question above about evaluating synthetic data quality.

You can also check for smoothness in the latent space by generating samples from interpolated points in the latent space and seeing if they produce smooth transitions in the generated data.

#### Can I use TimeVAE for forecasting?

Yes - you can use the underlying methodology - although we are not sure if it can be called TimeVAE in this context.

See these repositories for implementations focused on forecasting:

- https://github.com/readytensor/rt_forecasting_variational_encoder_pytorch
- https://github.com/readytensor/rt_forecasting_var_enc_fcst_w_pretraining

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact Information

For any inquiries or collaborations, please contact the lead author at: `<lead_author_first_name>.<lead_author_last_name>@gmail.com`.

See the paper for author details.
