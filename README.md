# PCG Signal Generation Using Diffusion Models: The DiffWave Approach
In this repository, you will find a Python implementation for synthesizing high-quality PCG signals using diffusion models. This work is based on [DiffWave](https://github.com/lmnt-com/diffwave/tree/master) repository. 
## Setup 
Create conda environment with the Python version of 3.8.5. Install the requirements and the DiffWave library as follows:

```
pip install diffwave
```
or 

```
git clone https://github.com/lmnt-com/diffwave.git
cd diffwave
pip install .
```
## DataSet 
The dataset used in this project was obtained from the [George B. Moody PhysioNet Challenge 2022](https://moody-challenge.physionet.org/2022/). Due to its large size, the data is not included in this repository. Due to its large size, the data is not included in this repository. To obtain the data, please run the following command:
```
wget -r -N -c -np https://physionet.org/files/circor-heart-sound/1.0.3/

```
> Dataloader -> gets the high quality normal PCG signals and saves them into anther directory

## Training and Evaluation 
1. Run the preprocess step required by the model:
  ```
  python -m diffwave.preprocess /path/to/dir/containing/wavs
  ```
2. To train the model, run the following command:
```
python __main__.py --model_dir /path/to/model/dir --data_dirs /path/to/dir/containing/wavs --max_steps 1000
```
3. Inference
  ```
  python -m diffwave.inference_all --fast /path/to/model --spectrogram_path /path/to/spectrogram  -o /path/to/output 
  ```
   > This code has been modified from the original version.

To monitor the learning progress, run the following command in another shell:
```
tensorboard --logdir /path/to/model/dir --bind_all

```
