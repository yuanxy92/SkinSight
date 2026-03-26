# SkinSight

Code and data for **SkinSight**.

This project is built upon the excellent work of [**VGGT-Long**](https://github.com/DengKaiCQ/VGGT-Long).  
We gratefully acknowledge the authors for making their code publicly available.

---

## Setup, Installation & Running

### 🖥️ 1. Hardware and System Environment

This project has been developed and tested under the following environment:

```
CPU(s): Intel Core i7-14700K / Intel Core i9-13900  
GPU(s): NVIDIA RTX 4090 (24 GiB VRAM)  
RAM: 64 GB  
```

---

### 📦 2. Environment Setup

#### Step 1: Create Environment & Install Dependencies

We recommend using `conda` (or `miniconda`) to create a clean environment:

```bash
conda create -n skinsight python=3.10.18
conda activate skinsight
```

Install PyTorch:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

#### Step 2: Download Pre-trained Weights

Download all required pre-trained weights  
(including VGGT, Pi3, and MapAnything by default):

```bash
bash ./scripts/download_weights.sh
```

---

#### Step 3 (Optional): Compile Loop-Closure Correction Module

```bash
python setup.py install
```

---

### 🚀 3. Running the Code

Run inference with:

```bash
python skinsight_recon.py --image_dir ./path_of_images
```

---

### ⚙️ Configuration

You can modify parameters in:

```
configs/base_config.yaml
```

To use a custom configuration file:

```bash
python skinsight_recon.py \
  --image_dir ./path_of_images \
  --config ./configs/base_config.yaml
```

---

## Notes

- Ensure all required weights are downloaded before running the code.
- For large-scale experiments, a GPU with sufficient VRAM (≥24 GB recommended) is preferred.
- Configuration files can be duplicated and modified to explore different parameter settings.

---

## Acknowledgement

This project builds upon **VGGT-Long**.  
We sincerely thank the original authors for their valuable contributions to the community.