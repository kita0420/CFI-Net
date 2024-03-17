# Cross-patch Feature Interaction Net with Edge Refinement for Retinal Vessel Segmentation

## This is an implementation of the CFI-Net.

## Environment Configuration：

```
* Python3.6
* Pytorch1.2.0
* CUDA 11.1
* Best trained with GPUs
```

## File structure：

```
  ├── datasets: The datasets we used in our paper
  ├── data_process:  Preprocessing method we used
  ├── net.method.py: The CFI-Net module code
  ├── net.network.py: The CFI-Net structure code
  ├── Constants.py:  parameter settings
  └── data_crop.read_data_crop.py: Preprocessing of data like random flipping and rotation
```
