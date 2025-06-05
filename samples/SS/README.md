# Super Sampling Workload 

Our super sampling workload uses Fourier-Based Super Resolution from https://github.com/iamxym/Deep-Fourier-based-Arbitrary-scale-Super-resolution-for-Real-time-Rendering

```
@inproceedings{zhang2024deep,
  title={Deep Fourier-based Arbitrary-scale Super-resolution for Real-time Rendering},
  author={Zhang, Haonan and Guo, Jie and Zhang, Jiawei and Qin, Haoyu and Feng, Zesen and Yang, Ming and Guo, Yanwen},
  booktitle={ACM SIGGRAPH 2024 Conference Papers},
  pages={1--11},
  year={2024}
}
```

## Instructions
All the necessary changes to apply ReFrame to FBSR are included in this folder. 
Our changes are labeled by `# > REFRAME start` and `# > REFRAME end` tags.
Please clone the original repository from the authors and update the included Python files to run FBSR with ReFrame. 

```bash
git clone git@github.com:iamxym/Deep-Fourier-based-Arbitrary-scale-Super-resolution-for-Real-time-Rendering.git fbsr

cp *.py fbsr/.
cd fbsr
```

After setting up the folder:

1. Follow the README instructions from the original authors to set up your test environment. 
2. Download the Sun Temple scene from the Unreal Engine FAB marketplace [here](https://fab.com/s/2d678819f47f).
3. Follow the instructions from the original authors to build the [modified Unreal Engine](https://github.com/fuxihao66/UnrealEngine/tree/5.1) and render the necessary frames.
4. Modify the paths in `configs.py` accordingly. Use this file to choose between using ReFrame and running the baseline.
5. Run `python3 inference.py` to run inference and collect output images.
