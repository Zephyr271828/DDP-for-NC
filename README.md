# DDP-for-NC

The concept of Neural Collapse (NC) derives from a 2020 paper called "Prevalence of neural collapse during the terminal phase of deep learning training" by Vardan Papyan et al. The authors of this paper also provided a source code to test this phenomenon with Resnet18 neural network, MNIST datatset, and MSE loss function. However, I found two problems with this code:

1. The code takes more than 4 hours to complete 350 epochs of training (even slower with more sophisticated datasets and larger neural networks) if I'm running it NYU's High-Performance Computer (HPC), making it inconvenient to run experiments under different settings
   
2. The code structure is somehow unordered, making it time-consuming to locate and modify specific parts of the code.

Therefore, I made two adjustments to the original code:
1. I integrated Distributed Data Parallel (DDP) to the original code so that multiple GPUs can be leveraged at the same time for the training, which improves the efficiency of the code by at most 4 times.
   
2. I moved all the functions and classes to ddp_utils.py to make the main python file more concise, making it easier to modify experimental settings and debug.

Some remaining problems are:
1. It seems DDP will influence the result of the experiments (e.g. more fluctuations in the experimental graphs). Some hyperparameter fine-tuning may be necessary to improve the performance, yet the influence of DDP on the final result is inevitable according to some DDP literature.

2. Multiple nodes may be leveraged so that we may adopt more than 4 GPUs to further improve the efficiency. Yet I haven't integrated that function in my code so the improvement is limited.
