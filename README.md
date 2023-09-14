# <picture> <source media="(prefers-color-scheme: dark)" srcset="assets/logo_w.png"> <img alt="Shows an illustrated sun in light color mode and a moon with stars in dark color mode." src="assets/logo_b.png">
 </picture>



The "Combining Active and Semi-Supervised Learning" internship proposes three general frameworks intended to provide users and the community with an idea of how to combine Active Learning and semi-supervised learning. For more detailed information about the frameworks, you can find it in the notebook folder [here](Notebook/frameworks.ipynb). Additionally, to understand how the code works, I have created a notebook that explains it as well, which you can find [here](Notebook/understand_code.ipynb).

> :bulb: Note: In this repository, the frameworks are applied to image classification tasks.


<!-- [Notebook](codes/pytorch_ml_project.ipynb) -->
## Actie learning and semi-supervised learning

In the table below, there are all methods of Active Learning and semi-supervised learning implemented in order to evaluate the frameworks
<!-- > :bulb: Note: The figure below represents a machine learning project in an academic setting. Therefore, the deployment part is missing. However, don't worry, we will cover it later. -->

| Semi-Supervised Learning                 | Active Learning                            |
|:---------------------:|:-------------------------------------:|
| FixMath [1]                      | least-confidence[4]|
| FlexMatch [2]                               | EntroSAmpling[4]    | 
| PseudoLabel [3]                               | Core-Set Approach[5]  |
|                               |  BatchBALD[6]|



## Usage

### Start with Docker
**Step1: Check your environment**

You need to properly install Docker and nvidia driver first. To use GPU in a docker containerYou also need to install nvidia-docker2 ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)).Then, Please check your CUDA version via `nvidia-smi`

**Step2: Clone the project**
```shell
git clone https://gitlab.multitel.be/signal/stages/stage2023/nasserali-combining-al-and-ssl.git
```
**Step3: Build the Docker image**
Before building the image, you may modify the [Dockerfile](Dockerfile) according to your CUDA version.The CUDA version we use is 11.7. You can change the base image tag according to [this site](https://hub.docker.com/r/nvidia/cuda/tags).You also need to change the `--extra-index-url` according to your CUDA version in order to install the correct version of Pytorch.You can check the url through [Pytorch website](https://pytorch.org).
Use this command to build the image
```shell 
cd nasserali-combining-al-and-ssl && docker build -t comb_ssl_Al .
```

## Example
```
python main.py --framework Framework1 --ALstrat  EntropySampling --SSLstrat fixmatch  --model VGG  --dataset svhn  --nStart 40 --nQuery 10  --nEnd 100 --n_epoch 5
```
It runs Combination experiment using ResNet18 and CIFAR-10 data, querying according to the EntropySampling algorithm for AL and  fixmatch . The result will be saved in the **./results** directory as "npy" files.

## References
[1] Sohn, K., Berthelot, D., Li, C. L., Zhang, Z., Carlini, N., Cubuk, E. D., ... & Raffel, C. (2020). FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. *arXiv preprint arXiv:2001.07685*.

[2]  Zhang, B., Wang, Y., Hou, W., Wu, H., Wang, J., Okumura, M., & Shinozaki, T. (2022). FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling. *arXiv preprint arXiv:2110.08263*.

[3] Lee, D.-H., & Others. (2013). Pseudo-label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks. In *Workshop on Challenges in Representation Learning, ICML* (Vol. 3, No. 2, pp. 896). Atlanta.

<!-- - [x] (PROXY, ICLR'20) Selection via Proxy: Efficient Data Selection for Deep Learning [paper](https://arxiv.org/pdf/1906.11829.pdf) [code](https://github.com/stanford-futuredata/selection-via-proxy) -->

[4] (Least Confidence/Margin/Entropy, IJCNN'14) A New Active Labeling Method for Deep Learning, IJCNN, 201.

[5] (CORE-SET, ICLR'18) Active Learning for Convolutional Neural Networks: A Core-Set Approach [paper](https://arxiv.org/pdf/1708.00489.pdf) [code](https://github.com/ozansener/active_learning_coreset).

[6]  (BatchBALD, NIPS'19) BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning [paper](https://papers.nips.cc/paper/2019/file/95323660ed2124450caaac2c46b5ed90-Paper.pdf) [code](https://github.com/BlackHC/BatchBALD)
