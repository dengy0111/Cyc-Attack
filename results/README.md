
# Reproducibility Data

This directory contains the data required to reproduce the experimental results presented in our paper.

## Directory Structure

- **`fig5/`**: Contains the adversarial upstream inputs used to generate the downstream trajectories for **Target 1**, as presented in **Figure 5**.
- **`fig6/`**: Contains the adversarial upstream inputs used to generate the downstream trajectories for **Target 2**, as presented in **Figure 6**.
- **`trajectory/`**: Contains the original forecast trajectories and the corresponding adversarial target trajectories used in our experiments.

### Adversarial Upstream Inputs (`fig5/` and `fig6/`)

Each directory contains NetCDF (`.nc`) files representing adversarially perturbed upstream inputs. These files are processed through the **GraphCast + TempestExtremes** pipeline to generate the corresponding downstream tropical cyclone trajectories.

Each `.nc` file follows the naming convention:

`(YYYY-MM-DDT0000, tgt*) adversarial_upstream_input.nc`

where:

- **`YYYY-MM-DDT0000`** denotes the initialization date and time of the corresponding forecasting experiment.
- **`tgt*`** identifies the adversarial target (e.g., `tgt1` or `tgt2`).
- **`adversarial_upstream_input`** indicates that the file contains an adversarially perturbed input to the upstream forecasting model.

### Original Forecast and Adversarial Target Trajectories (`trajectory/`)

The `trajectory/` directory contains:

- **Original forecast trajectories**: Tropical cyclone trajectories detected by **TempestExtremes** from **GraphCast** predictions generated using clean (unperturbed) inputs.
- **Adversarial target trajectories**: Target trajectories constructed for our adversarial attack experiments.

The adversarial target trajectory files correspond to the following experimental settings:

- **`submission_full`**: Corresponds to **Target 1** in the paper.
- **Other adversarial target trajectories**: Correspond to **Target 2** in the paper.
- **Hurricane Zeta (2020)**: In addition to the targets described above, this case study includes several additional adversarial target trajectories for further evaluation.

The original forecast trajectories serve as references, while the adversarial target trajectories specify the desired deviations from the original forecasts.

Together, these data support the reproduction and evaluation of the adversarial downstream tropical cyclone trajectories reported in the paper.
