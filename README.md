# Information Theoretic Representation Distillation
The provided code is for reproducing the CIFAR100 experimental results in our paper [Information Theoretic Representation Distillation
](https://arxiv.org/abs/2112.00459). The code is tested with Python 3.8.

## Running

1. Fetch the pretrained teacher models by:

```
sh scripts/fetch_pretrained_teachers.sh
```
which will download and save the models to `save/models`

2. Install dependancies:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tensorboard_logger
```


3. An example of performing distillation with ITRD losses is given as follows:
```
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_16_2 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0 --alpha_it 1.01
```

where the flags are explained as:
- `--path_t`: specify the path of the teacher model
- `--model_s`: specify the student model, see 'models/__init__.py' to check the available model types.
- `--b`: the weight of other distillation losses, default: `None`
- `--r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
- `--lambda_corr`: Correlation loss weighting
- `--lambda_mutual`: Mutual information loss weighting
- `--alpha_it`: Renyi's alpha-order used in the correlation loss

## Benchmark Results on CIFAR-100:
Performance is measured by classification accuracy (%)

1. Teacher and student are of the **same** architectural type.

| Teacher <br> Student | wrn-40-2 <br> wrn-16-2 | wrn-40-2 <br> wrn-40-1 | resnet56 <br> resnet20 | resnet110 <br> resnet20 | resnet110 <br> resnet32 | resnet32x4 <br> resnet8x4 |  vgg13 <br> vgg8 |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|:-----------:|
| Teacher <br> Student |    75.61 <br> 73.26    |    75.61 <br> 71.98    |    72.34 <br> 69.06    |     74.31 <br> 69.06    |     74.31 <br> 71.14    |      79.42 <br> 72.50     | 74.64 <br> 70.36 |
| KD | 74.92 | 73.54 | 70.66 | 70.67 | 73.08 | 73.33 | 72.98 |
| FitNet | 73.58 | 72.24 | 69.21 | 68.99 | 71.06 | 73.50 | 71.02 |
| AT | 74.08 | 72.77 | 70.55 | 70.22 | 72.31 | 73.44 | 71.43 |
| SP | 73.83 | 72.43 | 69.67 | 70.04 | 72.69 | 72.94 | 72.68 |
| CC | 73.56 | 72.21 | 69.63 | 69.48 | 71.48 | 72.97 | 70.71 |
| VID | 74.11 | 73.30 | 70.38 | 70.16 | 72.61 | 73.09 | 71.23 |
| RKD | 73.35 | 72.22 | 69.61 | 69.25 | 71.82 | 71.90 | 71.48 |
| PKT | 74.54 | 73.45 | 70.34 | 70.25 | 72.61 | 73.64 | 72.88 |
| AB | 72.50 | 72.38 | 69.47 | 69.53 | 70.98 | 73.17 | 70.94 |
| FT | 73.25 | 71.59 | 69.84 | 70.22 | 72.37 | 72.86 | 70.58 |
| FSP | 72.91 | N/A | 69.95 | 70.11 | 71.89 | 72.62 | 70.23 |
| NST | 73.68 | 72.24 | 69.60 | 69.53 | 71.96 | 73.30 | 71.53 |
| CRD | 75.48 | 74.14 | 71.16 | 71.46 | 73.48 | 75.51 | 73.94 |
| WCoRD | 76.11 | 74.72 | **71.92** | 71.88 | 74.20 | 76.15 | 74.72 |
| ReviewKD | **76.12** | 75.09 | 71.89 | - | 73.89 | 75.63 | 74.85 |
| **ITRD** | **76.12** | **75.18** | 71.47 | **71.99** | **74.26** | **76.69** | **74.93** |

2. Teacher and student are of **different** architectural type.

| Teacher <br> Student | vgg13 <br> MobileNetV2 | ResNet50 <br> MobileNetV2 | ResNet50 <br> vgg8 | resnet32x4 <br> ShuffleNetV1 | resnet32x4 <br> ShuffleNetV2 | wrn-40-2 <br> ShuffleNetV1 |
|:---------------:|:-----------------:|:--------------------:|:-------------:|:-----------------------:|:-----------------------:|:---------------------:|
| Teacher <br> Student |    74.64 <br> 64.60    |      79.34 <br> 64.60     |  79.34 <br> 70.36  |       79.42 <br> 70.50       |       79.42 <br> 71.82       |      75.61 <br> 70.50      |
| KD | 67.37 | 67.35 | 73.81 | 74.07 | 74.45 | 74.83 |
| FitNet | 64.14 | 63.16 | 70.69 | 73.59 | 73.54 | 73.73 |
| AT | 59.40 | 58.58 | 71.84 | 71.73 | 72.73 | 73.32 |
| SP | 66.30 | 68.08 | 73.34 | 73.48 | 74.56 | 74.52 |
| CC | 64.86 | 65.43 | 70.25 | 71.14 | 71.29 | 71.38 |
| VID | 65.56 | 67.57 | 70.30 | 73.38 | 73.40 | 73.61 |
| RKD | 64.52 | 64.43 | 71.50 | 72.28 | 73.21 | 72.21 |
| PKT | 67.13 | 66.52 | 73.01 | 74.10 | 74.69 | 73.89 |
| AB | 66.06 | 67.20 | 70.65 | 73.55 | 74.31 | 73.34 |
| FT | 61.78 | 60.99 | 70.29 | 71.75 | 72.50 | 72.03 |
| NST | 58.16 | 64.96 | 71.28 | 74.12 | 74.68 | 74.89 |
| CRD | 69.73 | 69.11 | 74.30 | 75.11 | 75.65 | 76.05 |
| WCoRD | 70.02 | 70.12 | 74.68 | 75.77 | 76.48 | 76.68 |
| ReviewKD | 70.37 | 69.89 | - | **77.45** | **77.78** | **77.14** |
| **ITRD** | **70.39** | **71.34** | **75.49** | 76.69 | 77.40 | 77.09 |

## Binary Distillation
Code can be found at: https://drive.google.com/file/d/1WJ_rGIsQ-SaqvXzNsfkxcNA6rWBbcnD8/view?usp=sharing
Relevant comments: https://github.com/roymiles/ITRD/issues/1

## Citation
```
@inproceedings{miles2022itrd,
  title={Information Theoretic Representation Distillation},
  author={Miles, Roy and Lopez-Rodriguez, Adrian and Mikolajczyk, Krystian},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2022}
}
```

## Acknowledgements
Our code is based on the code given in [RepDistiller](https://github.com/HobbitLong/RepDistiller).
