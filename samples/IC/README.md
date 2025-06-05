# Image Composition Workload 

Our image composition workload uses Implicit Depth from https://github.com/nianticlabs/implicit-depth

```
@inproceedings{watson2023implict,
  title={Virtual Occlusions Through Implicit Depth},
  author={Watson, Jamie and Sayed, Mohamed and Qureshi, Zawar and Brostow, Gabriel J and Vicente, Sara and Mac Aodha, Oisin and Firman, Michael},
  booktitle={Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
}
```

## Instructions
All the necessary changes to apply ReFrame to Implicit Depth are included in this folder. 
Our changes are labeled by `# > REFRAME start` and `# > REFRAME end` tags.
Please clone the original repository from the authors and update the included Python files to run Implicit Depth with ReFrame. 

```bash
git clone git@github.com:nianticlabs/implicit-depth.git

cp options.py implicit-depth/.
cp bd_model.py implicit-depth/experiment_modules/.
cp networks.py implicit-depth/modules/.
cp inference.py implicit-depth/inference/.

cd implicit-depth
```

After setting up the folder:

1. Follow the README instructions from the original authors to set up your test environment. 
    * We use the [HyperSim pretrained model](https://storage.googleapis.com/niantic-lon-static/research/implicit-depth/models/implicit_depth_temporal_hypersim.ckpt) in our experiments.
    * We use the example data for the [Garden Chair](https://storage.googleapis.com/niantic-lon-static/research/implicit-depth/example_data.zip) provided in the repo.
2. Follow the [steps for inference](https://github.com/nianticlabs/implicit-depth?tab=readme-ov-file#%EF%B8%8F-inference) to create the config, txt, and keyframe files as required.
3. Run inference using the following command:
    ```
    python3 -m inference.inference \
        --config configs/models/implicit_depth.yaml \
        --load_weights_from_checkpoint weights/implicit_depth_temporal_hypersim.ckpt \
        --data_config example_data/scans/config/config.yaml \
        --rendered_depth_map_load_dir example_data/renders \
        --output_base_path example_data/predictions/ \
        --dataset_path example_data/scans \
        --use_reframe
    ```
    * Remove the `--use_reframe` argument to run the  baseline model.
4. Follow the [steps for composition](https://github.com/nianticlabs/implicit-depth?tab=readme-ov-file#%EF%B8%8F-compositing) to create the final images using the implicit depth method.
