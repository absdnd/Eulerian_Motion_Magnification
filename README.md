# Python Implementation of Eulerian Motion Magnification

This repository contains a python3 implementation of the paper ["Eulerian Video Magnification for Revealing Subtle Changes in the World"
](http://people.csail.mit.edu/mrub/evm/).

## Dependencies

+ scipy
+ scikit-image
+ opencv==3.4.1.15
+ numpy
+ pyrtools==1.0.0

We recommend installing the dependencies using conda. 

## Execution
We support two execution butterworth bandpass filtering and ideal bandpass filtering. For evaluation, we utilize the videos provided the authors [here](http://people.csail.mit.edu/mrub/evm/#code). Place the videos in the `./source` folder. 

We allow for command line arguments in each of the codes. 

-  Quality factor `--Qf = {20,40,60,70,75,80,85,90}`
-  Stability index  `--index = {1 or all}`
-  Number of error images `--stack ={2,3}`
-  Run for all quality factors `--all_Q = {0,1}`
-  Number of repeated runs, `--runs = {1,2,3,4,5,6,7,8,9,10}`

## Live Magnification

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

### Citation 

If you utilize this code for your work, please cite the following paper. 
```@article{Wu12Eulerian,
  author = {Hao-Yu Wu and Michael Rubinstein and Eugene Shih and John Guttag and Fr\'{e}do Durand and
  William T. Freeman},
  title = {Eulerian Video Magnification for Revealing Subtle Changes in the World},
  journal = {ACM Transactions on Graphics (Proc. SIGGRAPH 2012)},
  year = {2012},
  volume = {31},
  number = {4},
}
```
