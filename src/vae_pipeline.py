import os, warnings
import numpy as np
import time


from data_utils import (
    load_yaml_file,
    load_data,
    split_data,
    scale_data,
    inverse_transform_data,
    save_scaler,
    save_data,
)
import paths
from vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    save_vae_model,
    get_posterior_samples,
    get_prior_samples,
    load_vae_model,
)
from visualize import plot_samples, plot_latent_space_samples, visualize_and_save_tsne


def run_vae_pipeline(dataset_name: str, vae_type: str):
    # ----------------------------------------------------------------------------------
    # Load data, perform train/valid split, scale data

    # read data
    data = load_data(data_dir=paths.DATASETS_DIR, dataset=dataset_name)

    # split data into train/valid splits
    train_data, valid_data = split_data(data, valid_perc=0.1, shuffle=True)

    # scale data
    scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)

    # ----------------------------------------------------------------------------------
    # Instantiate and train the VAE Model

    # load hyperparameters from yaml file
    hyperparameters = load_yaml_file(paths.HYPERPARAMETERS_FILE_PATH)[vae_type]

    # instantiate the model
    _, sequence_length, feature_dim = scaled_train_data.shape
    vae_model = instantiate_vae_model(
        vae_type=vae_type,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        **hyperparameters,
    )

    # train vae
    train_vae(
        vae=vae_model,
        train_data=scaled_train_data,
        max_epochs=1000,
        verbose=1,
    )

    # ----------------------------------------------------------------------------------
    # Save scaler and model
    model_save_dir = os.path.join(paths.MODELS_DIR, dataset_name)
    # save scaler
    save_scaler(scaler=scaler, dir_path=model_save_dir)
    # Save vae
    save_vae_model(vae=vae_model, dir_path=model_save_dir)

    # ----------------------------------------------------------------------------------
    # Visualize posterior samples
    x_decoded = get_posterior_samples(vae_model, scaled_train_data)
    plot_samples(
        samples1=scaled_train_data,
        samples1_name="Original Train",
        samples2=x_decoded,
        samples2_name="Reconstructed Train",
        num_samples=5,
    )
    # ----------------------------------------------------------------------------------
    # Generate prior samples, visualize and save them

    # Generate prior samples
    prior_samples = get_prior_samples(vae_model, num_samples=train_data.shape[0])
    # Plot prior samples
    plot_samples(
        samples1=prior_samples,
        samples1_name="Prior Samples",
        num_samples=5,
    )

    # visualize t-sne of original and prior samples
    visualize_and_save_tsne(
        samples1=scaled_train_data,
        samples1_name="Original",
        samples2=prior_samples,
        samples2_name="Generated (Prior)",
        scenario_name=f"Model-{vae_type} Dataset-{dataset_name}",
        save_dir=os.path.join(paths.TSNE_DIR, dataset_name),
        max_samples=2000,
    )

    # inverse transformer samples to original scale and save to dir
    inverse_scaled_prior_samples = inverse_transform_data(prior_samples, scaler)
    save_data(
        data=inverse_scaled_prior_samples,
        output_file=os.path.join(
            os.path.join(paths.GEN_DATA_DIR, dataset_name),
            f"{vae_type}_{dataset_name}_prior_samples.npz",
        ),
    )

    # ----------------------------------------------------------------------------------
    # If latent_dim == 2, plot latent space
    if hyperparameters["latent_dim"] == 2:
        plot_latent_space_samples(vae=vae_model, n=8, figsize=(15, 15))

    # ----------------------------------------------------------------------------------
    # later.... load model
    loaded_model = load_vae_model(vae_type, model_save_dir)

    # Verify that loaded model produces same posterior samples
    new_x_decoded = loaded_model.predict(scaled_train_data)
    print(
        "Preds from orig and loaded models equal: ",
        np.allclose(x_decoded, new_x_decoded, atol=1e-5),
    )

    # ----------------------------------------------------------------------------------


if __name__ == "__main__":
    # check `/data/` for available datasets
    dataset = "sine_subsampled_train_perc_20"

    # models: vae_dense, vae_conv, timeVAE
    model_name = "timeVAE"

    run_vae_pipeline(dataset, model_name)
