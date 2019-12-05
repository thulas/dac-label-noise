
# Deep Abstaining Classifier: Label denoising for  Deep Learning

PyTorch implementation of the deep abstaining classifier (DAC) from the  ICML 2019 paper:

**Combating Label Noise in Deep Learning Using Abstention**, Sunil Thulasidasan, Tanmoy Bhattacharya, Jeff Bilmes, Gopinath Chennupati, Jamaludin Mohd-Yusof 

The DAC uses an abstention loss function for identifying both arbitrary and systematic label noise  while training deep neural networks. 

## Identifying Systematic Label Noise

The DAC can be used to learn features or corrupting transformations that are associated with unreliable labels. As an example, in the "random monkeys" experiment,  all the monkey images in the train-set have their labels randomized. During prediction, the DAC abstains on most of the monkey images in the test-set. 

<p float="left">
<img src="imgs/monkey_tile.png" width="300" >
<img src="imgs/rand_monk_expt_dac_monk_dist.png" width="300">
</p>

In another experiment, we blur a subset (20%)  of the images in the training set and randomize their labels. The DAC learns to abstain on predicting on the blurred images during test time.


<p float="left">
  <img src="imgs/blurred_sample_tile_4x4.png" width="250" />
  <img src="imgs/blurred_expt_dac_blurred_val_pred_dist.png" width="300" /> 
  <img src="imgs/blurred_expt_dac_vs_dnn_val_acc2.png" width="300" />
</p>


To re-run the random monkeys experiment described in the paper, 

- download the STL-10 dataset from https://cs.stanford.edu/~acoates/stl10/
- copy `data/train_y_downshifted_random_monkeys.bin` to the STL-10 data directory
- copy `data/test_y_downshifted_random_monkeys.bin` to the STL-10 data directory

and then run as follows:


`python train_dac.py --datadir  <path-to-stl10-data> --dataset stl10-c --train_y train_y_downshifted_random_monkeys.bin --test_y test_y_downshifted_random_monkeys.bin --nesterov --net_type vggnet -use-gpu --epochs 200 --loss_fn dac_loss --learn_epochs 20 --seed 0`

In the above experiment, the best abstention occurs around epoch 75.


## Identifying Arbitrary Label Noise

The DAC can also be used to identify arbitrary label noise where there might not be an underlying corrupting feature or transformation, but classes get mislabeled with a certain probability. 

### Training Protocol

- Use DAC to identify label noise
- Eliminiate train samples that are abstained
- Retrain on cleaner set using regular cross-entropy loss

The DAC gives state-of-the-art results in label-noise experiments. 

<p float="left">
  <img src="imgs/cifar_10_60_ln.png" width="300" />
  <img src="imgs/cifar_100_60_ln.png" width="300" /> 
  <img src="imgs/cifar_10_80_ln.png" width="300" />
  <img src="imgs/webvision.png" width="300" />
</p>
[GCE: Generalized Cross-Entropy Loss (Zhang et al NIPS ‘18);  Forward (Patrini et al, CVPR ’17);  MentorNet (Li et al, ICML ‘18)]

More results are in our ICML 2019 paper. 

### Tested with:

- Python 2.7
- PyTorch 1.0.1

### Citation
```
@InProceedings{pmlr-v97-thulasidasan19a,
  title = 	 {Combating Label Noise in Deep Learning using Abstention},
  author = 	 {Thulasidasan, Sunil and Bhattacharya, Tanmoy and Bilmes, Jeff and Chennupati, Gopinath and Mohd-Yusof, Jamal},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {6234--6243},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California, USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v97/thulasidasan19a/thulasidasan19a.pdf},
```

This is open source software available under the BSD Clear license;

© (or copyright) 2019. Triad National Security, LLC. All rights reserved.
 
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

