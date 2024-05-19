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

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact Information

For any inquiries or collaborations, please contact the lead author at: `<lead_author_first_name>.<lead_author_last_name>@gmail.com`.

See the paper for author details.
