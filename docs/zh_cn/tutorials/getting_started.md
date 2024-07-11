## Detectron2 快速上手

本文将简要介绍 detectron2 内置命令行工具的使用方法。

有关如何使用 API 来进行实际编码的教程，
请参阅我们的[Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)，
其中详细介绍了如何使用现有模型进行推理，以及如何使用自定义数据集来训练内置模型。


### 使用预训练模型推理演示

1. 从[模型库](MODEL_ZOO.md)中选取一个模型及其配置文件，例如，`mask_rcnn_R_50_FPN_3x.yaml`。
2. 运行我们提供 `demo.py` 可以演示内置配置。运行指令如下：
```shell
cd demo/
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```
由于配置是针对训练设置的，因此我们需要通过设置 `MODEL.WEIGHTS` 来从模型库中指定一个模型来进行评估。
此命令将进行推理并在 OpenCV 窗口中显示可视化效果。

想要了解命令行参数的细节，可以参阅 `demo.py -h` 或是源码。 一些常见的参数如下：
* 想在 __网络摄像头__ 上运行，可将 `--input files` 替换为 `--webcam`。
* 想在 __视频__ 上运行，可将 `--input files` 替换为 `--video-input video.mp4`。
* 想在 __CPU__ 上运行，可在 `--opts` 参数之后添加`MODEL.DEVICE cpu`。
* 想将输出保存到目录（用于图像）或是文件（用于网络摄像头或者视频），可使用 `--output`。


### 使用命令行命令进行训练&评估

我们提供了"tools/plain_train_net.py" 和 "tools/train_net.py" 两个脚本，它们适用于 detectron2 中所有配置的训练。
你也可以以此作为参考来编写自己的训练脚本。

相比于 "train_net.py"， "plain_train_net.py" 支持的默认功能更少。但它包含的抽象也更精简，更加容易添加自定义逻辑。

要使用 "train_net.py" 训练模型，请先根据 [datasets/README.md](./datasets/README.md) 设置相应的数据集，然后运行：
```shell
cd tools/
./train_net.py --num-gpus 8 \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
```

以上配置是使用8个 GPU 来训练的。若想用单卡训练，你需要[修改一些参数](https://arxiv.org/abs/1706.02677)，例如：
```shell
./train_net.py \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

若要在评估模型的性能，可执行：
```shell
./train_net.py \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
有关更多选项，可以参阅 `./train_net.py -h`。

### 在代码中使用 Detectron2 API

参阅 [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
可以学习到如何使用 detectron2 API 来完成：
1. 在已有模型上完成推理
2. 在自定义数据集上训练内置模型

参阅 [detectron2/projects](https://github.com/facebookresearch/detectron2/tree/main/projects)，
了解使用 detctron2 来构建项目的更多方法。

