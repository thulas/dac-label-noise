
# Deep Abstaining Classifier: Label denoising for  Deep Learning

PyTorch implementation of the deep abstaining classifier (DAC) from the  ICML 2019 paper:

**Combating Label Noise in Deep Learning Using Abstention**, Sunil Thulasidasan, Tanmoy Bhattacharya, Jeff Bilmes, Gopinath Chennupati, Jamaludin Mohd-Yusof 

The DAC uses an abstention loss function for identifying arbitrary and systematic label noise  while training deep neural networks. For example, in the "random monkeys" experiment from the paper, all the monkey images in the train-set have their labels randomized. During prediction, the DAC abstains on all the monkey images in the test-set. 

<div class="row">
  <div class="column">
<img src="https://github.com/thulas/dac-label-noise/blob/master/imgs/monkey_tile.png" width="300" >
    </div>
  <div class="column">
<img src="https://github.com/thulas/dac-label-noise/blob/master/imgs/rand_monk_expt_dac_monk_dist.png" width="300">
 </div>
  </div>


To re-run the random monkeys experiment described in the paper, download the STL-10 dataset from https://cs.stanford.edu/~acoates/stl10/
and then run as follows:


`python train_dac.py --datadir  <path-to-stl10-data> --dataset stl10-c --train_y data/train_y_downshifted_random_monkeys.bin --nesterov --net_type vggnet -use-gpu --epochs 75 --loss_fn dac_loss --learn_epochs 20 --seed 0`



### Tested with:

- Python 2.7
- PyTorch 1.0.1



© (or copyright) 2019. Triad National Security, LLC. All rights reserved.
 
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

