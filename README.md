# Human Shape Estimator for Multi-Frame Inputs

This is the code for **Exploiting Synthetic Data for Human Shape Estimation on Multi-Frame Inputs**.

## Installation

Please unzip our code files and install the packages in `requirements.txt`.
Before installing the packages, we recommend installing Python 3.9.5, Pytorch 1.9.0, and torchvision 0.10.0.
Please install Pytorch compatible with your CUDA version.

```
conda create --name your_env_name python=3.9.5
conda activate your_env_name

pip install torch==1.9.0+cuXXX torchvision==0.10.0+cuXXX -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

To render 3D mesh, the OSMesa has to be installed with pyrender package.
Please go to [this website](https://pyrender.readthedocs.io/en/latest/install/index.html) and follow the instructions.

Next, you need to download the SMPL body model.
Please download the male and female models from [here](https://smpl.is.tue.mpg.de/), and the neutral model from [here](https://smplify.is.tue.mpg.de/).
After you register at both websites, please download the model files and locate them as below.

```
data
|-- smpl
    |-- basicModel_f_lbs_10_207_0_v1.0.0.pkl
    |-- basicModel_m_lbs_10_207_0_v1.0.0.pkl
    |-- basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

Now, you can preprocess these files by running the command below.

```
python3 utils/data_preprocessing.py
```

Before downloading image datasets, you have to write the path to your dataset directory in `utils/config.py`.
Please make your own path end with `/`.

```
BASE_DATASET_DIR = 'your/dataset/directory/'  # Write your/dataset/directory/
```

To evaluate our methods on the 3DPW dataset, you need to download the dataset from [here](https://virtualhumans.mpi-inf.mpg.de/3DPW/) and locate it in your dataset directory as below.

```
your/dataset/directory
|-- 3DPW
    |-- imageFiles
        |-- courtyard_arguing_00
        |-- courtyard_backpack_00
        |-- ...
    |-- sequenceFiles
        |-- test
        |-- train
        |-- validation
```

If you finish downloading the data, please move the whole files in `data/3DPW_infos/` into `your/dataset/directory/3DPW/infos/`.

```
mv data/3DPW_infos your/dataset/directory/3DPW/infos
```


## Running Demo Code

We provide the demo code to run our trained models on an input sequence in the 3DPW test protocol.
You can estimate the body shape from an image sequence of the given frame length and index.

```
python3 demo.py --n_frames=6 --test_index=2505
```

This code will make cropped input images, the ground truth mesh, meshes from the single-frame model, and a mesh from the multi-frame model with MPVEs in `examples/demo/`.


## Evaluation

You can also evaluate our models on the 3DPW test set or 3DPW test protocol.
We provide the model checkpoints of the single-frame and multi-frame models in `logs/`.
The test code will compute the average MPVE between the ground truth meshes and the mesh predictions.

```
# Evaluation of the single-frame method on the 3DPW test set
python3 test.py --model_dir=logs/single-frame_estimator_fine-tuned/model.pt --model_type=simple --state_dict_dir=logs/single-frame_estimator_fine-tuned/state_dict.pt

# Evaluation of the multi-frame method on the 3DPW test protocol (frame lengths from 1 to 15)
python3 test.py --model_dir=logs/multi-frame_estimator_fine-tuned/model.pt --model_type=shape_focused --state_dict_dir=logs/multi-frame_estimator_fine-tuned/state_dict.pt
```

We are preparing to make instructions for the SURREAL dataset downloading & preprocessing, the entire process from pre-training to fine-tuning with customized options.