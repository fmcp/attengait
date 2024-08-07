# AttenGait
AttenGait: when gait recognition meets attention and rich modalities

Support code for paper accepted for publication at Pattern Recognition. Note that some code parts are derived from [OpenGait](https://github.com/ShiqiYu/OpenGait)

## Abstract
Current gait recognition systems employ different types of manual attention mechanisms, like horizontal cropping of the input data to guide the training process and extract useful gait signatures for people identification. Typically, these techniques are applied using silhouettes as input, which limits the learning capabilities of the models. Thus, due to the limited information provided by silhouettes, state-of-the-art gait recognition approaches must use very simple and manually designed mechanisms, in contrast to approaches proposed for other topics such as action recognition. To tackle this problem, we propose AttenGait, a novel model for gait recognition equipped with trainable attention mechanisms that automatically discover interesting areas of the input data. AttenGait can be used with any kind of informative modalities, such as optical flow, obtaining state-of-the-art results thanks to the richer information contained in those modalities. We evaluate AttenGait on two public datasets for gait recognition: CASIA-B and GREW; improving the previous state-of-the-art results on them, obtaining 95.8% and 70.7% average accuracy, respectively.

## Code

### Requirements
Install the required libraries with:
```bash
pip install -r requirements.txt
```

### CASIA-B
#### 0. Data preparation
Download the CASIA-B dataset from [http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp) and extract the files. Raw RGB frames and silhouettes are required. Then, run the following code to prepare the data:
```python
python preprocessing/pretreatment_casiab_of.py --input_path DATAPATH/GaitDatasetB-silh/ --input_path_rgb DATAPATH/GaitDatasetB-RGB/ --output_path OUTPATH/CASIAB_of/ --worker_num 16
python preprocessing/generate_of_dataset_casiab.py --ofdir OUTPATH/CASIAB_of/ --outdir OUTPATH/CASIAB_of_dataset/
python preprocessing/generate_of_dataset_casiab_test.py --ofdir OUTPATH/CASIAB_of/ --outdir OUTPATH/CASIAB_of_dataset/ --mode ft
python preprocessing/generate_of_dataset_casiab_test.py --ofdir OUTPATH/CASIAB_of/ --outdir OUTPATH/CASIAB_of_dataset/ --mode test
```
#### 1. Training
```python
python mains/train.py --datadir=OUTPATH/CASIAB_of_dataset/ --experdir=EXPERPATH --nclasses=74 --epochs=8000 --extraepochs=1000 --pk --p 8 --k 8 --lr=0.0005 --attention_drop_rate=0.1 --softmax_attention --kernel_regularizer --prefix=attengait_casiab --lr_sched --cross_weight=1.0 --split_crossentropy --combined_output_length=32 --multi_gpu=2
```

#### 2. Testing
```python
python mains/test.py --datadir=OUTPATH/CASIAB_of_dataset/ --knn 1 --nclasses 50 --allcameras --model EXPERPATH/EXPERFOLDER/model-final.hdf5 --bs 1 --cross_weight=1.0 --split_crossentropy --softmax_attention --combined_output_length=32
```

### GREW
#### 0. Data preparation
Download the GREW dataset from [https://www.grew-benchmark.org/lander](https://www.grew-benchmark.org/lander) and extract the files. Then, run the following code to prepare the data:
```python
python preprocessing/pretreatment_grew_of.py --input_path DATAPATH/GREW/flow/train/ --output_path OUTPATH/GREW_of/train/ --worker_num 16 --mode train
python preprocessing/pretreatment_grew_of.py --input_path DATAPATH/GREW/flow/test/gallery/ --output_path OUTPATH/GREW_of/test/gallery/ --worker_num 16 --mode train
python preprocessing/pretreatment_grew_of.py --input_path DATAPATH/GREW/flow/test/probe/ --output_path OUTPATH/GREW_of/test/probe/ --worker_num 16 --mode test
python preprocessing/generate_of_dataset_grew.py --ofdir OUTPATH/GREW_of/train/ --outdir OUTPATH/GREW_of_dataset/
python preprocessing/generate_of_dataset_grew_test.py --ofdir OUTPATH/GREW_of/test/gallery/ --outdir OUTPATH/GREW_of_dataset/ --mode ft
python preprocessing/generate_of_dataset_grew_test.py --ofdir OUTPATH/GREW_of/test/probe --outdir OUTPATH/GREW_of_dataset/ --mode test
```

#### 1. Training
```python
python mains/train.py --datadir=OUTPATH/GREW_of_dataset/ --experdir=EXPERPATH --nclasses=20000 --epochs=2000 --extraepochs=1000 --pk --p 10 --k 4 --lr=0.00025 --attention_drop_rate=0.1 --softmax_attention --kernel_regularizer --prefix=attengait_grew --lr_sched --cross_weight=1.0 --split_crossentropy --combined_output_length=32 --multi_gpu=8
```

#### 2. Testing 
```python
python mains/test_grew_challenge.py --datadir=OUTPATH/GREW_of_dataset/ --knn 1 --nclasses 6000 --model EXPERPATH/EXPERFOLDER/model-final.hdf5 --bs 1 --cross_weight=1.0 --split_crossentropy --softmax_attention --combined_output_length=32
```

## References
Francisco M Castro, Rubén Delgado-Escaño, Ruber Hernández-García, Manuel J Marín-Jiménez, Nicolás Guil. _"AttenGait: Gait recognition with attention and rich modalities"_. Pattern Recognition, 2024
