# Python Implementation of Eulerian Motion Magnification

This repository contains a python implementation of the paper [Eulerian Video Magnification for Revealing Subtle Changes in the World
](http://people.csail.mit.edu/mrub/evm/).

# Environments and Dependenices

+ tensorflow = '1.12.0' (cudatoolkit=9.0 & cudnn=7.1.2)
+ scipy = '1.4.1'
+ sklearn = '0.22.1'
+ matplotlib = '3.1.3'
+ numpy = '1.18.1'

We recommend installing the dependencies using conda. 

# Getting Started
### Data Generation

+ Download the UCID dataset containing 1338 images .TIF images and place it in the `./data folder` in the root directory. Also download the [jpeg-read-toolbox](http://dde.binghamton.edu/download/jpeg_toolbox.zip) and place it in `/code/data_creation/dependencies/` as `jpeg_read_toolbox`. 

+ Create another folder called `all_needed_matlab_functions` in the folder `./code/data_creation/`. 

+ Then execute `./code/data_creation/data_maker.m` to create the Compressed_UCID_gray_full dataset. 

+ Execute  `./code/data_creation/patch_maker.m` , to create the training and testing 8 x 8 patches, with both `train = True` and `train = False`. 

+ After this please execute `./code/data_creation/save_error_images.m` in order to create all the error images for the generated patches. 

The file structure after saving the dataset would be as follows, 

```
/data
  /dataset
     /8
       /train
        /Quality_{Qf}
          /index_all
            /single
               /1
                 /single_error.mat
                 /single_dct_error.mat
               /2
                 ....
            /double
              /1
                 /double_error.mat
                 /double_dct_error.mat
              /2
                 ....
     
```

The dataset of error images is also available directly for download [here](https://drive.google.com/drive/folders/1nGSVn4so7GqcdH_4mHymqveYSWvuNriQ?usp=sharing), which is to be saved in `./data` folder in the root directory. 

### Training and Inference

The main code to run our approach is MCCNN_Classifier.py, which use the default runtime settings of our model as mentioned in the paper. They include, 

#### Command Line Parameters: 

These parameters can be supplied at the time of running the code. Below are the possible combinations that can be given for command line arguments. 

-  Quality factor `--Qf = {20,40,60,70,75,80,85,90}`
-  Stability index  `--index = {1 or all}`
-  Number of error images `--stack ={2,3}`
-  Run for all quality factors `--all_Q = {0,1}`
-  Number of repeated runs, `--runs = {1,2,3,4,5,6,7,8,9,10}`

#### Pretrained Models.

- For direct inference the pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1bpR2UoW7VyibSNFcQynlm_dnK1ITGGSi?usp=sharing).
- To use each quality factor separately download and place it under `proposed_results/MCNN/index_all`. 
- Alternatively, you can download all pretrained models by unzipping  the entire folder in `proposed_results/MCNN`.


#### Sample Command. 

To reproduce our results at `--Qf = 60`, `--index = 'all'` and `--stack = 3` for `--runs=10` the code is: 
```shell
python /code/generate_results/MCCNN.py \
-- Qf = 60 \
-- index = all\
-- stack = 3\
-- runs = 10\
-- ptr = 1
```
- To execute for all quality factors please set the parameter `--all_Q=1`
- Set `--ptr = 1` to use the pretrained model. 

#### Obtaining Results 

The results are saved in the following directory structure, with `results_itr_~.mat` containing the resultant predictions. 

```
/proposed_results
  /MCNN
     /index_all
         /Quality_~
            /runs
              /MCNN_stack_3_scale_1
                  /results/
                      /results_itr_0.mat
                      ..
                      ..
                      ..
                  /loss_plots/
                      /loss_itr_0.png
                      ..
                      ..
                      ..
                  /weights/
                     /best_weights_itr_0.mat
                     ..
                     ..
                     ..
```

### References: 

- Schaefer, Gerald, and Michal Stich. ["UCID: An uncompressed color image database."](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/5307/0000/UCID-an-uncompressed-color-image-database/10.1117/12.525375.short) Storage and Retrieval Methods and Applications for Multimedia 2004. Vol. 5307. International Society for Optics and Photonics, 2003.

- 

### Citation 

```@inproceedings{Harish2020Double,
  title={Double JPEG Compression Detection for Distinguishable Blocks in Images Compressed with Same Quantization Matrix},
  author={Abhinav Narayan Harish*, Vinay Verma*, Nitin Khanna},
  booktitle={IEEE International Workshop on Machine Learning for Signal Processing}
  year={2020}
```
