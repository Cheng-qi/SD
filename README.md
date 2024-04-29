#  A Universal Strategy for Smoothing Deceleration in Deep Graph Neural Networks
This repository is the implementation of the experiments in the following paper:

```
@article{RN13,
   author = "Cheng, Qi and Long, lang and Feng, Weixing",
   title = "A Universal Strategy for Smoothing Deceleration in Deep Graph Neural Networks",
   year = "2024",
   type = "Journal Article"
}
```

## Quick Start

### Requirements
   - torch 2.2.0+cu118
   - torch-cluster 1.6.3+pt22cu118
   - torch-geometric 2.5.1
   - torch-scatter 2.1.2+pt22cu118
   - torch-sparse 0.6.18+pt22cu118
   - torch-spline-conv 1.2.2+pt22cu118
   - scikit-learn 0.24.2
   - numpy 1.20.3
   - CUDA 11.8
Install related dependencies:
```
pip install -r requirements.txt
```
## Run examples
```
bash ./run.sh
```
