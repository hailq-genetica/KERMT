<h1 align="center">KERMT</h1>

This is the official code repository for the paper titled [Multitask finetuning and acceleration of chemical pretrained models for small molecule drug property prediction](https://arxiv.org/abs/2510.12719).

<p align="center">
    <img width="750" src="figures/concept.png"/>
</p>


**K**inetic GROV**ER** **M**ulti-**T**ask (KERMT) is a pretrained graph neural network model for molecular property prediction.

KERMT is an enhanced reimplementation of the [GROVER](https://arxiv.org/abs/2007.02835) model. The KERMT implementation uses PyTorch Distributed Data Parallel (DDP) for distributed pretraining,  automates hyperparameter tuning, and accelerates finetuning and prediction using [cuik-molmaker](https://github.com/NVIDIA-Digital-Bio/cuik-molmaker).

This implementation is based on the [original GROVER implementation](https://github.com/tencent-ailab/grover) and [paper](https://arxiv.org/abs/2007.02835).

## Requirements
We recommend using a Docker container for running the model. For developers, we have provided a Dockerfile that was used to create the container.

## Setup

#### Clone the repository
```bash
git clone https://github.com/hailq-genetica/KERMT.git
cd KERMT
```

#### Pretrained Model Download

##### KERMT v2.0 (recommended)

The released KERMT v2.0 model is hosted on Hugging Face: [**nvidia/NV-KERMT-70M-v2**](https://huggingface.co/nvidia/NV-KERMT-70M-v2). Its contrastive pretraining is described in the Contrastive KERMT preprint ([arXiv:2606.11508](https://arxiv.org/abs/2606.11508)). The repository bundles the pretrained hybrid checkpoint (`kermt_contrastive_v2.0.pt`) together with its vocabulary files (`pretrain_atom_vocab.json`, `pretrain_bond_vocab.json`, `pretrain_smiles_vocab.pkl`), distributed under the NVIDIA Open Model License. The vocabulary files are an inseparable part of the model — keep them alongside the checkpoint.

Download the full bundle into a local directory:
```bash
pip install huggingface_hub
huggingface-cli download nvidia/NV-KERMT-70M-v2 --local-dir model/NV-KERMT-70M-v2
```

#### Build the container
```bash
docker build --rm -t kermt:latest -f Dockerfile .
```

#### Run the container with GPUs
```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ./:/code -v ./data:/data -v ./model:/model -v ./runs:/runs -it --name kermt  kermt:latest
```

```bash
source /softwares/miniconda3/etc/profile.d/conda.sh && conda activate kermt
cd code
```

## ADMET Fine tuning
These scripts finetune and evaluate the released checkpoint on the [Therapeutics Data Commons](https://tdcommons.ai/) `admet_group` benchmarks — single-task, across 5 seeds on TDC's official splits — and report mean±std in each task's official TDC metric (AUROC/AUPRC for classification, MAE/Spearman for regression) next to published baselines (MolE and TxGemma-27B). They rely on `PyTDC`, which is included in the container environment (`environment.yml`); the TDC splits download automatically into the `--tdc_path` directory on first run.

#### Classification Tasks
```
python scripts/kermt_admet_group_cls.py --code_dir /code --ckpt /model/NV-KERMT-70M-v2/kermt_contrastive_v2.0.pt --tdc_path /data --seeds 1 2 3 4 5 --epochs 50
```

#### Regression Tasks
```
python scripts/kermt_admet_group_reg.py --code_dir /code --ckpt /model/NV-KERMT-70M-v2/kermt_contrastive_v2.0.pt --tdc_path /data --seeds 1 2 3 4 5 --epochs 50
```

Each run prints a summary table and writes a `results.json` (mean, std per task) under `/runs/admet_group_{cls,reg}`. Pass `--only <Benchmark_Name ...>` to run a subset, or `--dry_run` to print the commands without training.

## Prediction
A finetuned model can be used to make predictions on target molecules. The finetuned model is saved in this directory: `/runs/admet_group_cls/seed1/{task_name}/ckpt_link/model.pt`.

#### Prediction with Finetuned Model
``` bash
python main.py predict \
    --data_path tests/data/finetune/test.csv \
    --checkpoint_dir /runs/admet_group_cls/seed1/AMES/ckpt_link/ \
    --no_features_scaling \
    --features_generator rdkit_2d_normalized_cuik_molmaker \
    --output path/to/predictions.csv
```

## Hardware Requirements
- GPUs are required for pretraining, finetuning, and prediction. Multiple GPUs can be used for distributed pretraining. NVIDIA GPUs with atleast 32GB of vRAM and Volta or newer architectures is recommended.


## References
- Paper:[Multitask finetuning and acceleration of chemical
pretrained models for small molecule drug
property prediction](https://arxiv.org/abs/2510.12719)
- Contrastive KERMT (v2.0) preprint: [arXiv:2606.11508](https://arxiv.org/abs/2606.11508)
- Dataset: [Figshare link](https://figshare.com/articles/dataset/Datasets_for_Multitask_finetuning_and_acceleration_of_chemical_pretrained_models_for_small_molecule_drug_property_prediction_/30350548/2)
- [Original GROVER paper](https://arxiv.org/abs/2007.02835)
- [Original GROVER implementation](https://github.com/tencent-ailab/grover)
