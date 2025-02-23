# TimePFN
Official repository for "TimePFN: Effective Multivariate Time Series Forecasting with Synthetic Data" (AAAI 2025).

This repository contains the codebase of the TimePFN. We recommend using a conda virtual environment to load the dependencies listed in `requirements.txt`.

We provide the model checkpoint, testing, training, and fine-tuning scripts. Please check `pfn_scripts`. For the datasets, please refer to iTransformer's `datasets.zip` [gdrive link](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=sharing).

Download them and put them under the directory `./datasets`.

To generate synthetic datasets for the pretraining task, please refer to the directory `synthetic_data_generation`. We have already provided the default values for the hyperparameters; however, feel free to experiment. Please read the comments and directives in the bash scripts.

