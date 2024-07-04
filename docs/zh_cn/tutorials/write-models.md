# 编写模型

如果您正在尝试做一些全新的事情,比如完全从头实现一个模型.此时,
您可能对修改或者扩展现有模型的一些组件感兴趣.为此,
我们还提供了允许用户重写标准模型中某些内部组件的机制.


## 注册新组件

对于诸如"特征提取主干网络","任务头"等用户常想自定义的概念,我们提供了注册机制,
它可以注入这些自定义实现,并支持通过配置文件进行配置.

例如添加一个新的主干网络,可以将以下代码导入到您的代码中:
```python
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class ToyBackbone(Backbone):
  def __init__(self, cfg, input_shape):
    super().__init__()
    # create your own backbone
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=16, padding=3)

  def forward(self, image):
    return {"conv1": self.conv1(image)}

  def output_shape(self):
    return {"conv1": ShapeSpec(channels=64, stride=16)}
```

在这段代码中,我们按照 [Backbone](../modules/modeling.html#detectron2.modeling.Backbone) 类的接口实现一个新的主干网络,
作为 `Backbone` 的子类,将其注册到 [BACKBONE_REGISTRY](../modules/modeling.html#detectron2.modeling.BACKBONE_REGISTRY)中.
导入此代码后，detectron2 可以将类的名称链接到其实现。因此，您可以编写以下代码：

```python
cfg = ...   # read a config
cfg.MODEL.BACKBONE.NAME = 'ToyBackbone'   # or set it in the config file
model = build_model(cfg)  # it will find `ToyBackbone` defined above
```

再举一个例子,要为 Generalized R-CNN 元架构中的 ROI 头添加新的能力,
您可以实现一个新的 [ROIHeads](../modules/modeling.html#detectron2.modeling.ROIHeads) 的子类并将其注册到 `ROI_HEADS_REGISTRY` 中.
[DensePose](../../../projects/DensePose)
和 [MeshRCNN](https://github.com/facebookresearch/meshrcnn) 就是这样的两个例子,它们实现了新的 ROI 头来执行新的任务.
[projects/](../../../projects/) 包含了更多这样实现不同架构模型的例子.

在 [API documentation](../modules/modeling.html#model-registries) 中可以找到完整的注册表列表.
您可以在这些注册表中注册组件，以自定义模型的不同部分或整个模型.

## 使用显式参数构建模型

Registry 是将配置文件中的名称连接到实际代码的桥梁.它们旨在涵盖用户经常需要更换的几个主要组件.
但是,基于文本的配置文件的功能有时是有限的,一些更深层次的定制可能只有通过编写代码才能获得.

detectron2 中的大多数模型组件都有一个清晰 `__init__` 的接口,用于记录它需要的输入参数.
使用自定义参数调用它们将为您提供模型的自定义变体。

例如,在 Faster R-CNN 的回归框头上使用 __自定义损失函数__,我们可以执行以下操作:

1. 损失目前在 [FastRCNNOutputLayers](../modules/modeling.html#detectron2.modeling.FastRCNNOutputLayers) 中计算.
我们需要实现一个包含自定义损失函数的变体或者子类,这里不妨叫 `MyRCNNOutput`.

2. 使用 `box_predictor=MyRCNNOutput()` 作为参数而非 `FastRCNNOutputLayers` 的默认设置来调用 `StandardROIHeads`.
若其他参数保持不变,则通过[可配置的`__init__`](../modules/config.html#detectron2.config.configurable) 机制来轻松实现:

   ```python
   roi_heads = StandardROIHeads(
     cfg, backbone.output_shape(),
     box_predictor=MyRCNNOutput(...)
   )
   ```

3. (可选) 如果我们想从配置文件中启用这个新模型,则需要注册:

   ```python
   @ROI_HEADS_REGISTRY.register()
   class MyStandardROIHeads(StandardROIHeads):
     def __init__(self, cfg, input_shape):
       super().__init__(cfg, input_shape,
                        box_predictor=MyRCNNOutput(...))
   ```
