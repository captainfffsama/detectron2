## Detectron2快速上手

本文将简要介绍 detectron2 内置命令行工具的使用方法.

有关如何使用 API 来进行实际编码的教程,
请参阅我们的[Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5),
其中详细介绍了如何使用现有模型进行推理,以及如何使用自定义数据集来训练内置模型.


### 使用预训练模型推理演示

1. 从[模型库](MODEL_ZOO.md)中选取一个模型及其配置文件,例如,`mask_rcnn_R_50_FPN_3x.yaml`.
2. 运行我们提供 `demo.py` 可以演示内置配置.运行指令如下:
```shell
cd demo/
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```
由于配置是针对训练设置的,因此我们需要通过设置 `MODEL.WEIGHTS` 来从模型库中指定一个模型来进行评估.
此命令将进行推理并在 OpenCV 窗口中显示可视化效果.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.


### Training & Evaluation in Command Line

We provide two scripts in "tools/plain_train_net.py" and "tools/train_net.py",
that are made to train all the configs provided in detectron2. You may want to
use it as a reference to write your own training script.

Compared to "train_net.py", "plain_train_net.py" supports fewer default
features. It also includes fewer abstraction, therefore is easier to add custom
logic.

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](./datasets/README.md),
then run:
```
cd tools/
./train_net.py --num-gpus 8 \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
```

The configs are made for 8-GPU training.
To train on 1 GPU, you may need to [change some parameters](https://arxiv.org/abs/1706.02677), e.g.:
```
./train_net.py \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

To evaluate a model's performance, use
```
./train_net.py \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
For more options, see `./train_net.py -h`.

### Use Detectron2 APIs in Your Code

See our [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
to learn how to use detectron2 APIs to:
1. run inference with an existing model
2. train a builtin model on a custom dataset

See [detectron2/projects](https://github.com/facebookresearch/detectron2/tree/main/projects)
for more ways to build your project on detectron2.
