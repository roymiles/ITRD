# ITRD
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


3. The experiments can be run with the following commands:

#### Same architecture experiments
    python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill itrd --model_s wrn_16_2 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0 --alpha_it 1.01
    
    python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill itrd --model_s wrn_40_1 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0 --alpha_it 1.01
    ​
    python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill itrd --model_s resnet20 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0 --alpha_it 1.01
    ​
    python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill itrd --model_s resnet20 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0 --alpha_it 1.01
    ​
    python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill itrd --model_s resnet32 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0 --alpha_it 1.01
    ​
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill itrd --model_s resnet8x4 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0 --alpha_it 1.01
    ​
    python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill itrd --model_s vgg8 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0 --alpha_it 1.01

#### Cross architectural experiments
    python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill itrd --model_s MobileNetV2 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0 --alpha_it 1.5
    ​
    python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill itrd --model_s MobileNetV2 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0 --alpha_it 1.5
    ​
    python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill itrd --model_s vgg8 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0 --alpha_it 1.5
    ​
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill itrd --model_s ShuffleV1 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0 --alpha_it 1.5
    ​
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill itrd --model_s ShuffleV2 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0 --alpha_it 1.5
    ​
    python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill itrd --model_s ShuffleV1 -b 1.0 -r 1.0 --lambda_corr 2.0 --lambda_mutual 1.0  --alpha_it 1.5


## Citation
```
@article{Miles2021itrd,
author = {Miles, Roy and Rodriguez, Adrian Lopez and Mikolajczyk, Krystian},
title = {{Information Theoretic Representation Distillation}},
year = {2021}
}
```

## Acknowledgements
Our code is based on the code given in [RepDistiller](https://github.com/HobbitLong/RepDistiller).
