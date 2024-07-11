# 使用内置数据集

访问 [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog) 可以获取数据集中数据，
通过访问 [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) 则可以获取其元数据（比如类名等）。
本文将介绍如何设置内置数据集，以便通过上述 API 来使用它们。
[使用自定义数据集](https://detectron2.readthedocs.io/tutorials/datasets.html) 将更加深入地探讨来如何使用 `DatsetCatalog` 和 `MetadataCatalog`，
以及如何向它们添加新数据集

Detectron2 内置了对一些数据集的支持。
假定这些数据集存在在由环境变量 `DETECTRON2_DATASETS` 指定的目录下。
在此目录下， detectron2 将在必要时按照以下目录结构来搜寻数据集。
```
$DETECTRON2_DATASETS/
  coco/
  lvis/
  cityscapes/
  VOC20{07,12}/
```

您可以通过设置环境变量 `export DETECTRON2_DATASETS=/path/to/datasets` 的方式来设置内置数据集所在的位置。
若为设置，则内置数据集目录为相对于您当前工作目录的 `./datasets`。

[model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) 中包含了使用这些内置数据的模型和配置。

## 用于 [COCO 实例/关键点检测](https://cocodataset.org/#download) 的数据集结构

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # 对应的 json 中提及的图片文件将存在此
```

您也可以使用 2014 版本的数据集。

一些内置测试 (`dev/run_*_tests.sh`) 使用 COCO 数据集的迷你版本，
您可以使用 `./datasets/prepare_for_tests.sh` 来下载它们。

## 用于 PanopticFPN 的数据集结构

从 [COCO 网站](https://cocodataset.org/#download) 提取全景标注，形成以下结构：
```
coco/
  annotations/
    panoptic_{train,val}2017.json
  panoptic_{train,val}2017/  # png 标注
  panoptic_stuff_{train,val}2017/  # 使用以下脚本生成
```

通过以下命令可以安装 panopticapi：
```shell
pip install git+https://github.com/cocodataset/panopticapi.git
```
然后， 运行 `python datasets/prepare_panoptic_fpn.py`， 从全景标注中提取语义标注。

## 用于 [LVIS 实例分割](https://www.lvisdataset.org/dataset) 的数据集结构

```
coco/
  {train,val,test}2017/
lvis/
  lvis_v0.5_{train,val}.json
  lvis_v0.5_image_info_test.json
  lvis_v1_{train，val}.json
  lvis_v1_image_info_test{,_challenge}.json
```

通过以下命令可以安装 lvis-api：
```shell
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

要使用 LVIS 标注来评价在 COCO 数据集上训练的模型，
运行 `python datasets/prepare_cocofied_lvis.py` 可以获得 "coco 形式的" LVIS 标注。

## 用于 [cityscapes](https://www.cityscapes-dataset.com/downloads/) 的数据集结构：

```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    # 以下是生成的 Cityscapes 全景标注
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
```
通过以下命令可以安装 cityscapesScripts：
```shell
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

注意： 要创建 labelTrainIds.png， 请先将目录结构整理成上述结构，然后运行 cityscapesScript ：
```shell
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```
实例分割不需要这些文件。

注意： 要生成 Cityscapes 全景分割数据集，请使用以下命令运行 cityscapesScript：
```shell
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createPanopticImgs.py
```
语义分割和实例分割不需要这些文件。

## 用于 [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/index.html) 的数据集结构
```
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
      # train.txt or val.txt, if you use these splits
  JPEGImages/
```

## 用于 [ADE20k 场景解析](http://sceneparsing.csail.mit.edu/) 的数据集结构
```
ADEChallengeData2016/
  annotations/
  annotations_detectron2/
  images/
  objectInfo150.txt
```
运行 `python datasets/prepare_ade20k_sem_seg.py` 可以生成目录 `annotations_detectron2`。
