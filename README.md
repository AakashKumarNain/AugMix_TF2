# AugMix

[Augmix](https://arxiv.org/pdf/1912.02781.pdf) is a new a data processing technique that mixes augmented images and enforces
consistent embeddings of the augmented images, which results in increased robustness and improved uncertainty calibration. This
technique achieves much better results as compared to other augmentation techniques. Not only it imporoves the accuracy of the
models but also contributes in improving the robustness of the models.

The [official code](https://github.com/google-research/augmix) is in PyTorch. This is a just a port from PyTorch to Tensorflow 2.0 for the same work. I used `ResNet20` as an example for the model but you can use whatever model you like.

## Requirements
Python>=3.6.x<br>
numpy>= 1.17<br>
Pillow>=6.2<br>
tensorflow==2.0<br>

## Usage:
I always recommend using `anaconda` for managing your environments but you can use `virtualenv` as well or you can directly install packages. It's your choice.

1. Clone the repo.
2. `pip install requirements.txt`
3. `python main.py --batch_size=128 --epochs=100`

## TODO:
- [x] Make everything modular
- [x] Custom **Early Stopping** and **History** for custom training loops
- [ ] Distributed training


## Citation
```
@article{hendrycks2020augmix,
  title={{AugMix}: A Simple Data Processing Method to Improve Robustness and Uncertainty},
  author={Hendrycks, Dan and Mu, Norman and Cubuk, Ekin D. and Zoph, Barret and Gilmer, Justin and Lakshminarayanan, Balaji},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2020}
}
```
