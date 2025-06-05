# Frame Extrapolation Workload 

Our frame extrapolation workload uses ExtraNet from https://github.com/fuxihao66/ExtraNet

```
@article{guo2021extranet,
    author = {Guo, Jie and Fu, Xihao and Lin, Liqiang and Ma, Hengjun and Guo, Yanwen and Liu, Shiqiu and Yan, Ling-Qi},
    title = {ExtraNet: Real-Time Extrapolated Rendering for Low-Latency Temporal Supersampling},
    year = {2021},
    month = {dec},
    journal = {ACM Trans. Graph.},
    volume = {40},
    number = {6},
    articleno = {278}
}
```

## Instructions
All the necessary changes to apply ReFrame to ExtraNet are included in this folder. 
Our changes are labeled by `# > REFRAME start` and `# > REFRAME end` tags.
Please clone the original repository from the authors and update the included Python files to run ExtraNet with ReFrame. 

```bash
git clone git@github.com:fuxihao66/ExtraNet.git

cp *.py ExtraNet/Model/.
cd ExtraNet
```

After setting up the folder:

1. Follow the README instructions from the original authors to set up your test environment. 
2. Download the Sun Temple scene from the Unreal Engine FAB marketplace [here](https://fab.com/s/2d678819f47f).
3. Follow the instructions from the original authors to build the [modified Unreal Engine](https://github.com/fuxihao66/UnrealEngine/tree/5.1) and render the necessary frames.
4. Follow the instructions from the original authors to pre-process the rendered buffers.
5. Update the `path` on Line 47 in `inference.py` to the location of the pre-processed buffers. 
6. Run `python3 inference.py` to run inference and collect output images.