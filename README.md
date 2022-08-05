# Human Shape Estimator for Multi-Frame Inputs

This is the code for **Exploiting Synthetic Data for Human Shape Estimation on Multi-Frame Inputs**.

## Installation

Please unzip our code files and install the packages in `requirements.txt`.
Before installing the packages, we recommend installing Python 3.9.5, Pytorch 1.9.0, and torchvision 0.10.0.
Please install Pytorch compatible with your CUDA version.

```
conda create --name your_env_name python=3.9.5
conda activate your_env_name

# Write your CUDA version (+cuXXX) or use CPU only (+cpu)
pip install torch==1.9.0+cuXXX torchvision==0.10.0+cuXXX -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

To render 3D mesh, the OSMesa has to be installed with pyrender package.
Please go to [this website](https://pyrender.readthedocs.io/en/latest/install/index.html) and follow the instructions.


## Data Preparation

First, you need to download the SMPL body model.
Please download the male and female models from [here](https://smpl.is.tue.mpg.de/), and the neutral model from [here](https://smplify.is.tue.mpg.de/).
After you register at both websites, please create `data/smpl/` folder and download the model files in the folder as follows:

```
data
|-- smpl
    |-- basicModel_f_lbs_10_207_0_v1.0.0.pkl
    |-- basicModel_m_lbs_10_207_0_v1.0.0.pkl
    |-- basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
|-- options
|-- ...
```

Next, you have to write the path to your dataset directory in `utils/config.py`.
Please make your own path end with `/`.

```
BASE_DATASET_DIR = 'your/dataset/directory/'  # Write your/dataset/directory/
```

To evaluate our methods on the 3DPW dataset, you need to download the dataset from [here](https://virtualhumans.mpi-inf.mpg.de/3DPW/) and locate it in your dataset directory `your/dataset/directory/`.

If you want to pre-train our models from scratch or evaluate on the SURREAL dataset, you need to visit [this website](https://github.com/gulvarol/surreal) and request access to the data.
Please follow the instructions in the website and download the dataset in your dataset directory `your/dataset/directory`.
The entire dataset requires large disk space, so you can download the partial dataset of videos (`*.mp4`), data infos (`*_info.mat`), and segmentations (`*_segm.mat`).

The structure of your dataset directory should be as follows:

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
|-- SURREAL (optional)
    |-- data
        |-- cmu
            |--test
            |--train
            |--val
```

Now, you can preprocess all data by running the command below.

```
python3 utils/data_preprocessing.py
```


## Running Demo Code

We provide the demo code to run our trained models on an input sequence in the 3DPW test protocol.
You need to download our test protocols and checkpoints of the single-frame and multi-frame models from [logs](https://drive.google.com/file/d/1SM_R1kEJ1wiqThgvsYm_Hl1v49aBz9aD/view?usp=sharing) and [protocol](https://drive.google.com/file/d/1d-r3G6L14j-KLHFh3CIrs2cWtWqVODbM/view?usp=sharing).
Please create `logs/` and `data/protocol/` folders and locate the downloaded files as follows:

```
data
|-- protocol
    |-- test_protocol_3dpw.npz
    |-- test_protocol_surreal.npz
|-- ...
logs
|-- multi-frame_estimator_fine-tuned
|-- multi-frame_estimator_pre-trained
|-- single-frame_estimator_fine-tuned
|-- single-frame_estimator_pre-trained
|-- splitted_simple_model_with_nonflip_augmentation (pre-trained backbone)
```

You can now estimate the body shape from an image sequence of given frame length and index.
This command will make cropped input images, the ground truth mesh, meshes from the single-frame model, and a mesh from the multi-frame model with MPVEs in `examples/demo/`.

```
python3 demo.py --n_frames=6 --test_index=2505  # input sequence length 6 and index 2505
```


## Evaluation

You can also evaluate our models on the 3DPW or SURREAL dataset.
The test code below will compute the average MPVE between the ground truth meshes and the mesh predictions.

```
# Evaluation of the pre-trained single-frame model on the SURREAL test set
python3 test.py --model_dir=logs/single-frame_estimator_pre-trained/model.pt --model_type=simple --state_dict_dir=logs/single-frame_estimator_pre-trained/state_dict.pt --dataset=surreal

# Evaluation of the pre-trained multi-frame model on the SURREAL test protocol (frame lengths from 1 to 15)
python3 test.py --model_dir=logs/multi-frame_estimator_pre-trained/model.pt --model_type=shape_focused --state_dict_dir=logs/multi-frame_estimator_pre-trained/state_dict.pt --dataset=surreal

# Evaluation of the fune-tuned single-frame model on the 3DPW test set
python3 test.py --model_dir=logs/single-frame_estimator_fine-tuned/model.pt --model_type=simple --state_dict_dir=logs/single-frame_estimator_fine-tuned/state_dict.pt --dataset=3dpw

# Evaluation of the fune-tuned multi-frame model on the 3DPW test protocol (frame lengths from 1 to 15)
python3 test.py --model_dir=logs/multi-frame_estimator_fine-tuned/model.pt --model_type=shape_focused --state_dict_dir=logs/multi-frame_estimator_fine-tuned/state_dict.pt --dataset=3dpw

# Evaluation of your model; Please check the options in test.py.
python3 test.py --model_dir=your/trained/model.pt --model_type=your_model_type --state_dict_dir=your/trained/state_dict.pt --dataset=target_dataset
```


## Training

We pre-train our models on the SURREAL dataset and then fine-tune them on the 3DPW dataset.
You can train your own models (single-frame and multi-frame) with the default option and the recommended options we provide.

```
# Pre-train a single-frame model.
python3 train.py --model=simple  # default option

# Fine-tune a single-frame model.
python3 train.py --model=simple --single_finetuning_options --checkpoint=your/pre-trained/model/state_dict.pt  # recommanded option

# Pre-train a multi-frame model.
python3 train.py --model=aggregation --multi_pretraining_options  # recommanded option

# Fine-tune a multi-frame model.
python3 train.py --model=aggregation --multi_finetuning_options --checkpoint=your/pre-trained/model/state_dict.pt  # recommanded option
```

If you want to customize your own model, please check the detailed options in `utils/options.py`.
The logs files (checkpoints, tensorboard summary, comments, options) will be saved in `logs/` folder.
You can check your training summaries by running tensorboard.

```
tensorboard --logdir=logs/your_trained_log_folder/
```