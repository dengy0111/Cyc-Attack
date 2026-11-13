
# Adversarial Attacks on Downstream Weather Forecasting Models: Application to Tropical Cyclone Trajectory Prediction

**Authors:** Yue Deng, Francisco Santos, Pang-Ning Tan, and Lifeng Luo

**Paper:** [arXiv:2510.10140](https://arxiv.org/abs/2510.10140)

## Overview

We propose adversarial attack methods that construct carefully manipulated weather inputs to deep learning weather forecasting (DLWF) models, with the goal of misleading downstream tropical cyclone (TC) trajectory predictions. Specifically, our attacks aim to suppress or misdirect the originally predicted TC trajectory while inducing a false alarm corresponding to a fabricated trajectory steered away from the original forecast.

Our implementation employs **GraphCast**, developed by Google DeepMind, as the DLWF model and **TempestExtremes** as the downstream TC detection and tracking system. Together, they form an end-to-end forecasting pipeline for evaluating how adversarial perturbations to upstream weather inputs propagate to downstream TC trajectory predictions.
