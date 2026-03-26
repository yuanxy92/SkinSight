


This repository contains the source code for our work:

[VGGT-Long: Chunk it, Loop it, Align it, Pushing VGGT's Limits on Kilometer-scale Long RGB Sequences]

based on XXX

##  Setup, Installation & Running

### 🖥️ 1 - Hardware and System Environment 

This project was developed, tested, and run in the following hardware/system environment

```
Hardware Environment：
    CPU(s): Intel Core i7-14700K / Intel Core i9-13900
    GPU(s): NVIDIA RTX 4090 (24 GiB VRAM)
    RAM: 64.0 GiB 
```

### 📦 2 - Environment Setup 

#### Step 1: Dependency Installation

Creating a virtual environment using conda (or miniconda),

```cmd
conda create -n skinsight python=3.10.18
conda activate skinsight
```

Next, install `PyTorch`,

```cmd
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 
```

Install other requirements,

```cmd
pip install -r requirements.txt
```

#### Step 2: Weights Download

Download all the pre-trained weights needed(Download weights for VGGT, Pi3, and MapAnything by default.):

```cmd
bash ./scripts/download_weights.sh
```

#### Step 3 (Optional) : Compile Loop-Closure Correction Module

```cmd
python setup.py install
```

### 🚀 3 - Running the code


```cmd
python vggt_long_new.py --image_dir ./path_of_images
```

You can modify the parameters in the `configs/base_config.yaml` file. If you have created multiple yaml files to explore the effects of different parameters, you can specify the file path by adding `--config` to the command. For example:

```cmd
python vggt_long.py --image_dir ./path_of_images --config ./configs/base_config.yaml
```

