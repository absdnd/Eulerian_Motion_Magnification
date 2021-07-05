


# Python Implementation of Eulerian Motion Magnification

This repository contains a python3 implementation of the paper ["Eulerian Video Magnification for Revealing Subtle Changes in the World"
](http://people.csail.mit.edu/mrub/evm/).

https://user-images.githubusercontent.com/30770447/124435789-f298dc80-dd92-11eb-9a9c-ec8c81e09a97.mp4


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

-  Lower cutoff frequency `--fl`
-  Upper cutoff frequency `--fh`
-  lambda_c `--lambda_c ={2,3}`
-  Number of pyramid levels `--nLevels`

## Live Magnification

The Eulerian Motion Magnification code can also be used for real-time performance using only the intensity channel. We use butterworth bandpass filtering for real-time perfomance. Please run live_magnification.sh for real-time performance. 

<!-- ```shell
python /code/generate_results/MCCNN.py \
-- Qf = 60 \
-- index = all\
-- stack = 3\
-- runs = 10\
-- ptr = 1
``` -->

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
