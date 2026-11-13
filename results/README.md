# Reproducibility Data

This directory contains the data required to reproduce the experimental results presented in our paper.

## Directory Structure

- **`fig5/`**: Contains the adversarial upstream inputs used to generate the downstream trajectories for **Target 1**, as presented in **Figure 5**.
- **`fig6/`**: Contains the adversarial upstream inputs used to generate the downstream trajectories for **Target 2**, as presented in **Figure 6**.

Each directory contains NetCDF (`.nc`) files representing adversarially perturbed upstream inputs. These files are processed through the **GraphCast + TempestExtremes** pipeline to generate the corresponding downstream tropical cyclone trajectories.

## File Naming Convention

Each `.nc` file follows the naming convention:

`(YYYY-MM-DDT0000, tgt*) adversarial_upstream_input.nc`

where:

- **`YYYY-MM-DDT0000`** denotes the initialization date and time of the corresponding forecasting experiment.
- **`tgt*`** identifies the adversarial target (e.g., `tgt1` or `tgt2`).
- **`adversarial_upstream_input`** indicates that the file contains an adversarially perturbed input to the upstream forecasting model.

These files enable the reproduction of the adversarial downstream trajectories reported in Figures 5 and 6 of the paper.
