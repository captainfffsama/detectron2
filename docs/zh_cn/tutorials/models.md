# 使用模型

## 从 Yacs 配置中构建模型
使用诸如 `build_model`， `build_backbone`， `build_roi_heads` 等函数可以
从 yacs 配置对象中构建模型（及其子模型）:

```python
from detectron2.modeling import build_model
model = build_model(cfg)  # returns a torch.nn.Module
```

`build_model` 仅构建模型并使用随机参数填充权重。
请参阅下文了解如何将现有存档（checkpoint） 加载到模型以及如何使用 `model` 对象。

### 加载/保存存档（checkpoint）
```python
from detectron2.checkpoint import DetectionCheckpointer
DetectionCheckpointer(model).load(file_path_or_url)  # load a file, usually from cfg.MODEL.WEIGHTS

checkpointer = DetectionCheckpointer(model, save_dir="output")
checkpointer.save("model_999")  # save to output/model_999.pth
```

Detectron2 的存档管理器（checkpointer） 可以识别 pytorch 中的 `.pth` 格式和我们模型库中的 `.pkl` 为文件。
更改使用信息，参见 [API 文档](../modules/checkpoint.html#detectron2.checkpoint.DetectionCheckpointer)。

使用 `torch.{load,save}` 可以对 `.pth` 文件进行编辑，`.pkl` 文件则是使用 `pickle.{dump,load}` 进行操作。

### 使用模型

模型可以通过 `outputs =  model(inputs)` 来调用，其中 `inputs` 类型是 `list[dict]`。
每个字典对应一张图片，其所需要的键则取决于模型的类型，以及模型是处于训练模式还是评估模式。
例如，为了进行推理，所有现有模型都需要 `image` 键，以及可选的 “height” 和 “width”。
现有模式的输入及输出的详细格式解释如下。


__训练__: 在训练模式下，所有模型都需要在 `EventStorage` 的上下文管理器中使用。
训练的统计数据将被记录到存储 （storage） 中：
```python
from detectron2.utils.events import EventStorage
with EventStorage() as storage:
    losses = model(inputs)
```

__推理__：若您只是想使用现有模型进行简单的推理，
[DefaultPredictor](../modules/engine.html#detectron2.engine.defaults.DefaultPredictor)
是一个提供此类基本功能的模型装饰器。它对单个图像执行的默认行为包括模型加载，预处理。
具体请参阅使用文档。

如下，您也可以直接运行推理：
```python
model.eval()
with torch.no_grad():
    outputs = model(inputs)
```

### 模型输入格式

用户可以实现支持任何任意输入格式的自定义模型。
在这里，我们描述了 Detectron 2 中所有内置模型支持的标准输入格式。
它们都以 list[dict] 作为输入。
每个字典对应于一个图像的信息。

字典可能会包含以下键：

* “image”： 形状为 (C, H, W) 的 `Tensor` 类型。`cfg.INPUT.FORMAT` 定义了每个通道的含义。
  若可以的话，图像归一化将使用 `cfg.MODEL.PIXEL_{MEAN,STD}` 在模型内部进行。
* “height”，“width”： **推理**中**需要**的输出的高宽，不一定与 `image` 字段中的高宽相同。
  比如图像预处理中包含了图像尺寸调整，那么 `image` 字段中保存的就是进行尺寸调整之后的图像。
  但您可能想按照**原始分辨率**来输出。
  若本参数提供，模型将以本参数的分辨率来进行输出，而非使用 `image` 字段分辨输出。输入分辨率判定同理。
* “instances”： 用于训练的[Instances](../modules/structures.html#detectron2.structures.Instances)
  对象，其字段如下：
  + “gt_boxes”：[Boxes](../modules/structures.html#detectron2.structures.Boxes) 类型的对象，包含了 N 个包围框，与实例一一对应。
  + “gt_classes”： long 类型的 `Tensor`，是包含了 N 个标签的向量，值范围是 [0， 类别数目)。
  + “gt_masks”： [PolygonMasks](../modules/structures.html#detectron2.structures.PolygonMasks)
    或 [BitMasks](../modules/structures.html#detectron2.structures.BitMasks) 对象，包含了 N 个掩码，与实例一一对应。
  + “gt_keypoints”：[Keypoints](../modules/structures.html#detectron2.structures.Keypoints)
    对象，包含了 N 个关键点集合，与实例一一对应。
* “sem_seg”： 形状为（H，W）的 `Tensor[int]`。它是语义分割的标注答案掩码。其中的值表示从0开始的类别标签。
* “proposals”： [Instances](../modules/structures.html#detectron2.structures.Instances)
  对象，仅用于 Fast R-CNN 样式的模型训练，包含以下字段：
  + “proposal_boxes”： [Boxes](../modules/structures.html#detectron2.structures.Boxes) 对象，包含了 P 个待选框。
  + “objectness_logits”： `Tensor`, 包含了 P 个得分的向量，数量与待选框一一对应。

对于内置模型的推断，只需要“image”键，“width/height”是可选的。

我们目前没有为全景分割训练定义标准输入格式，因为模型现在使用自定义数据加载器生成的自定义格式。

#### 链接到数据加载器：

默认 [DatasetMapper](../modules/data.html#detectron2.data.DatasetMapper) 的输出是一个符合上述格式的字典。
在数据加载器执行数据加载之后，它变成了内置模型支持的 `list[dict]` 。

### 模型输出格式

当处于训练模式时，内置模型输出一个 `dict[str->ScalarTensor]`，其中包含了所有的 loss。

当处在推理模式，内置模型输出 `list[dict]`，其中元素数目和图像数目一致。
依据模型正在执行的任务，每个字典可能包含以下字段：

* “instances”： [Instances](../modules/structures.html#detectron2.structures.Instances)
  对象，包含以下字段
  * “pred_boxes”： [Boxes](../modules/structures.html#detectron2.structures.Boxes) 类型的对象，包含了 N 个预测框，与实例一一对应。
  * “scores”： `Tensor`，包含了 N 个置信度的向量。
  * “pred_classes”： `Tensor`，包含了 N 个标签的向量，值范围 [0， 类别数目)。
  + “pred_masks”： 形状为（N，H，W）的 `Tensor`, 表示了每个预测实例的掩码。
  + “pred_keypoints”： 形状为（N，关键点数目，3）的 `Tensor`。
    最后一个维度中的每一行都是（x，y，score）。置信度得分大于0。
* “sem_seg”： 形状为（类别数，H，W）的 `Tensor`，代表语义分割的预测。
* “proposals”： [Instances](../modules/structures.html#detectron2.structures.Instances)
  对象，包含以下字段：
  * “proposal_boxes”： [Boxes](../modules/structures.html#detectron2.structures.Boxes)
    对象，包含 N 个待选框。
  * “objectness_logits”： 包含 N 个置信度分数的 torch 向量。
* “panoptic_seg”： `(pred: Tensor, segments_info: Optional[list[dict]])` 的元组。
  `pred` 张量形状为（H，W），包含了每个像素的语义 id。

  * 若 `segments_info` 存在，每个字典描述了 `pred` 中的一个分割结果 id，且包含以下字段：

    * “id”： 分割结果 id
    * “isthing”： 分割结果是事物（thing） 还是填充（stuff）
    * “category_id”： 该分割结果的类别 id

    若一个像素的 id 并不存在与 `segments_info` 中，则其表示[全景分割](https://arxiv.org/abs/1801.00868)
    中定义的空标签。

  * 若 `segments_info` 为 None， `pred` 所有像素值必须 ≥ -1.
    值为 -1 的像素被视为空标签。
    否则，每个像素的类别 id 通过以下公式获得，
    `category_id = pixel // metadata.label_divisor`.


### 部分执行模型：

有时候你可能想获取模型内部的一个中间张量，比如某个层的输入，后处理之前的输出。
由于通常有数百个中间张量，因此没有一个API可以为您提供所需的中间结果。您有以下选项：

1. 编写一个（子）模型。按照[本教程](./write-models.md)，您可以重写模型组件（例如模型的头部），使其与现有组件做相同的事情，但返回您需要的输出。
2. 部分执行模型。您可以像往常一样创建模型，但使用自定义代码来执行它，而不是使用它的 `forward()` 。
   例如，下面的代码在掩码头之前获取掩码特征。

   ```python
   images = ImageList.from_tensors(...)  # preprocessed input tensor
   model = build_model(cfg)
   model.eval()
   features = model.backbone(images.tensor)
   proposals, _ = model.proposal_generator(images, features)
   instances, _ = model.roi_heads(images, features, proposals)
   mask_features = [features[f] for f in model.roi_heads.in_features]
   mask_features = model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])
   ```

3. 使用 [forward hooks](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks)。
   前向钩子可以帮助你获取某个模块的输入或输出。如果它们不是你想要的，至少可以和部分执行一起使用来获得其他张量。

以上所有选项都要求您阅读现有模型的文档，
有时还需要阅读现有模型的代码来理解内部逻辑，以便编写代码来获取内部张量。
