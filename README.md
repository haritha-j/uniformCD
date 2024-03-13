# UniformCD


This repository contains experiments related to the uniform CD objective function.


## Requirements

This code has been tested with;
- python == 3.9
- pytorch == 1.13
- cuda == 11.6

The primary requisites are;
- open3D
- 5py
- matplotlib
- numpy
- ifcopenshell
- sklearn
- scipy
- [chamferdist](https://github.com/krrish94/chamferdist)
- [emd](https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd)

## Notes

UniformCD loss is defined in the `calc_uniform_chamfer_loss_tensor` function in `src/chamfer.py`. 

The VRC implementation from [Density Aware Chamfer Distance](https://github.com/wutong16/Density_aware_Chamfer_Distance) is included in this comparison.

This loss has been tested with [PointSWD](https://github.com/VinAIResearch/PointSWD), [PointAttn](https://github.com/ohhhyeahhh/PointAttN), [InfoCD](https://github.com/Zhang-VISLab/NeurIPS2023-InfoCD), [hyperCD](https://github.com/Zhang-VISLab/HyperCD) and [Parameter prediction](https://github.com/haritha-j/industrial-facility-relationships).

[Chamferdist](https://github.com/krrish94/chamferdist) and [EMD](https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd) are used for computing losses. [PointFlowRender](https://github.com/zekunhao1995/PointFlowRenderer) is used for visualisation.



### VRC Model training and evaluation
  + To train a model: run `python train.py ./cfgs/*.yaml`, for example:
```
python train.py ./cfgs/vrc_plus.yaml
```
  + To test a model: run `python train.py ./cfgs/*.yaml --test_only`, for example:
```
python train.py ./cfgs/vrc_plus_eval.yaml --test_only
```

For additional instructions including setup and dataset download, please follow the instructions [here.](Density_aware_Chamfer_Distance/README.md) 

### Notebooks

This repo contains three notebooks which contain experiments related to visualisations of various loss functions and their correspondences, sphere morphing, and various visualisations of results from completion and reconstruction. Further information is provided within the notebooks.
