# EasyCounting
EasyCounting is an open source crowd/vehicle counting toolbox based on PyTorch.

## Features
- Easy plug-in. Enables you to run different models under a unified framework.
- Configurations.

## Requirements
More details in `reqirement.txt`. 

## Quickstart

1. Append the project's path to `PYTHONPATH` and 
   make sure to use absolute imports.
```shell
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project"
```

2. Preprocess the dataset.
```shell
# Take ShanghaiTech Part_A for example.
python datasets/SHHA/generate_dataset.py --data-root ~/workspace/datasets/ShanghaiTech/part_A_final
```

<details>
<summary>Arguments of <code>generate_dataset.py</code></summary>

- `--data-root`:
   - Path to the raw dataset downloaded from the Internet. 
   - Default: `~/workspace/datasets/{DATASET_NAME}`.
- `--destination`:
   - Where the preprocessed data will be saved in.
   - Default: `{PROJECT_PATH}/processed_data/{DATASET_NAME}`.
- `--resize-shape`:
   - Usage: `--resize-shape {width} {height}`.
   - Default: `None` (no resizing).

</details>

3. Edit the configuration (if needed) and then enjoy training your network!
```shell
# Like, train CSRNet on ShanghaiTech Part_A dataset.
python train.py --config configs/CSRNet/SHHA.py
```

## Supported methods and datasets

### Methods

|  Method  | Year | Venue |            Experiments            | Source Code                                                                   |
|:--------:|:----:|:-----:|:---------------------------------:|-------------------------------------------------------------------------------|
|  CSRNet  | 2018 |  CVPR |  [Link](models/CSRNet/README.md)  | [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch)       |
| DM_Count | 2020 |  NIPS | [Link](models/DM_Count/README.md) | [cvlab-stonybrook/DM-Count](https://github.com/cvlab-stonybrook/DM-Count)     |
|   PSNet  | 2020 | arXiv |   [Link](models/PSNet/README.md)  | [daimuuc/PyramidScaleNetwork](https://github.com/daimuuc/PyramidScaleNetwork) |
|  STDNet  | 2021 |  TMM  |  [Link](models/STDNet/README.md)  | [stk513486/STDNet](https://github.com/stk513486/STDNet)                       |

### Datasets
1. Crowd counting datasets 

|       Dataset       | Year |  Attributes | Avg. <br>Resolution | No.<br>Samples | No.<br>Instance | Avg.<br>Count | Source                                                                                                                          |
|:-------------------:|:----:|:-----------:|:-------------------:|:--------------:|:---------------:|:-------------:|---------------------------------------------------------------------------------------------------------------------------------|
|  Fudan-ShanghaiTech | 2019 |    Video    |      1080×1920      |     15,000     |     394,081     |       27      | [Homepage](https://github.com/sweetyy83/Lstn_fdst_dataset)                                                                      |
|        Venice       | 2019 |    Video    |       720×1280      |       167      |        -        |       -       | [Homepage](https://github.com/weizheliu/Context-Aware-Crowd-Counting)                                                           |
|       UCF-QNRF      | 2018 |  Congested  |      2013×2902      |      1,535     |    1,251,642    |      501      | [Homepage](https://www.crcv.ucf.edu/data/ucf-qnrf/)                                                                             |
| ShanghaiTech Part_A | 2016 |  Congested  |       589×868       |       482      |     241,677     |      501      | [Homepage](https://github.com/desenzhou/ShanghaiTechDataset) <br> [Kaggle](https://www.kaggle.com/datasets/tthien/shanghaitech) |
| ShanghaiTech Part_B | 2016 | Free Scenes |       768×1024      |       716      |      88,488     |      123      | [Homepage](https://github.com/desenzhou/ShanghaiTechDataset) <br> [Kaggle](https://www.kaggle.com/datasets/tthien/shanghaitech) |
|         MALL        | 2012 |    Video    |       480×640       |      2,000     |      62,325     |       31      | [Homepage](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)                                                  |
|         UCSD        | 2008 |    Video    |       158×238       |      2,000     |      49,885     |       25      | [Homepage](http://visal.cs.cityu.edu.hk/downloads/ucsdpeds-vids/)                                                               |

2. Vehicle counting datasets

| Dataset | Year |     Attributes    | Avg. <br>Resolution | No.<br>Samples | No.<br>Instance | Avg.<br>Count |  Source  |
|:-------:|:----:|:-----------------:|:-------------------:|:--------------:|:---------------:|:-------------:|:--------:|
|  UAVCC  | 2020 |     Drone-view    |       540×1024      |       885      |        -        |       -       |     -    |
|  CARPK  | 2017 |     Drone-view    |       720×1080      |      1,448     |        -        |       -       | [Homepage](https://lafi.github.io/LPN/) |
| TRANCOS | 2015 | Surveillance-view |      2013×2902      |      1,244     |      46,796     |      37       | [Homepage](https://gram.web.uah.es/data/datasets/trancos/index.html) |


## Acknowledgement
This project was inspired by:
- [gjy3035/C-3-Framework](https://github.com/gjy3035/C-3-Framework): An open-source PyTorch code for crowd counting
- [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab Semantic Segmentation Toolbox and Benchmark

Some codes were borrowed from:
- [gjy3035/C-3-Framework](https://github.com/gjy3035/C-3-Framework)
- [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch)
- [cvlab-stonybrook/DM-Count](https://github.com/cvlab-stonybrook/DM-Count)
- [daimuuc/PyramidScaleNetwork](https://github.com/daimuuc/PyramidScaleNetwork)
- [stk513486/STDNet](https://github.com/stk513486/STDNet)

Many thanks!
