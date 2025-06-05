# ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering

[Lufei Liu](https://www.lufei.ca), [Tor M. Aamodt](https://people.ece.ubc.ca/~aamodt/)

University of British Columbia (UBC) 


![ICML Logo](./docs/images/ICML-logo.svg#gh-light-mode-only)
![ICML Logo](./docs/images/ICML-logo-dark.svg#gh-dark-mode-only)

*To appear in ICML 2025*

---

## Lay Summary
Realistic visuals make video games and virtual reality feel more immersive and exciting, but creating these images can be slow and power-intensive. In fact, the better the image, the longer it takes. While animated movies can spend hours producing each frame, interactive experiences need to respond instantly to user input to feel smooth and believable.

Our research aims to make rendering faster, so we can save power and improve image quality without slowing down the system. We noticed that many frames displayed back-to-back look very similar, which inspired us to introduce a mechanism that only partially updates the neural networks involved in creating each frame. We strategically save parts of the neural network and reuse them for as long as possible before they start noticeably compromising the image.

Our technique accelerates the neural networks behind the visuals, cutting energy use and making it easier for less powerful hardware to keep up without sacrificing quality.



### More details coming soon!
Visit our [website](https://ubc-aamodt-group.github.io/reframe-layer-caching/) for updates.


## ReFrame 

To apply ReFrame to your target network, identify the most suitable concatenation in the network and inject the layer caching code. 
An example using a U-Net can be found in [./samples/unet/unet.py](./samples/unet/unet.py)

### Frame Extrapolation Sample
Our frame extrapolation workload uses [ExtraNet](). The modified source code can be found in [./samples/FE](./samples/FE/) with detailed instructions.

### Supersampling Sample
Our supersampling workload uses [Fourier-Based Super Resolution](https://github.com/iamxym/Deep-Fourier-based-Arbitrary-scale-Super-resolution-for-Real-time-Rendering). 
The modified source code can be found in [./samples/SS](./samples/SS/) with detailed instructions.

### Image Composition Sample


### FLIP Image Metric
Instructions to compute FLIP scores can be found [here](https://github.com/NVlabs/flip).

A sample script is included in [./samples/flip/compare_images.py](./samples/flip/compare_images.py).

Install FLIP with:
```bash
python3 -m pip install flip_evaluator
```


## Cite ReFrame
```
@inproceedings{liu2025reframe,
    title={ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering},
    author={Lufei Liu and Tor M. Aamodt},
    booktitle={Proceedings of International Conference on Machine Learning (ICML)},
    year={2025},
    organization={PMLR},
}
```