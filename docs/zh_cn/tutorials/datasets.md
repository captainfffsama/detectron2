## 使用自定义数据集

本文将解释数据集 API
（[DatasetCatalog](../modules/data.html#detectron2.data.DatasetCatalog)， [MetadataCatalog](../modules/data.html#detectron2.data.MetadataCatalog)）
是如何工作的，以及如何使用它们来添加自定义数据集。

[内置数据集](builtin_datasets.md) 列出 detectron2 支持的所有内置数据集。
如果您想要使一个自定义数据集可以复用 detectron2 的数据加载，
你需要完成以下事情：

1. __注册__ 您的数据集（即，告诉 detectron2 如何获取您的数据集）。
2. （可选）， 为您的数据集 __注册元数据__。

接下来，我们将详细解释上述两个概念。

[Colab 教程](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) 上有一个实时实例，
说明如何在自定义格式的数据集上注册和训练。

### 注册数据集

为了让 detectron2 知道如何获取名为 "my_dataset" 的数据集，
用户需要实现一个函数，该函数返回数据集中的项，然后将此函数告诉 detectron2：
```python
def my_dataset_function():
  ...
  return list[dict] in the following format

from detectron2.data import DatasetCatalog
DatasetCatalog.register("my_dataset", my_dataset_function)
# later, to access the data:
data: List[Dict] = DatasetCatalog.get("my_dataset")
```

这里的代码片段将名为 "my_dataset" 的数据集与返回数据的函数相关联。
如果多次调用，该函数必须返回相同的数据（具有相同的顺序）。
注册将一直有效，直到进程退出。

该函数可以执行任意操作，并应以下其中之一的格式，返回 `list[dict]` 中的数据：
1. Detectron2 的标注数据集字典，详情参见下文。这样它可以复用 detectron2 中大量的其他内置功能，建议尽量使用这种方式。
2. 任意自定义格式。您可以以任意格式返回任意字典，比如为新任务添加额外的键。但您还需要在下游正确的处理它们。详情参见下文。

#### 标准数据集字典

对于标准任务（实例检测，实例分割/语义分割/全景分割，关键点检测），
我们将原始数据集以类似 COCO 标注的格式加载到 `list[dict]` 中。
这是我们定义的数据集标准表示形式。

每个字典都包含一个图像有关信息。
字典可能含有以下字段，
必填字段根据数据加载器或任务的需求而各有不同（请参阅下文）。

```eval_rst
.. list-table：：
  ：header-rows： 1

  * - 任务
    - 字段
  * - 一般任务
    - file_name， height， width， image_id

  * - 实例检测/分割
    - annotations

  * - 语义分割
    - sem_seg_file_name

  * - 全景分割
    - pan_seg_file_name， segments_info
```

+ `file_name`: 图像文件的完整路径。
+ `height`, `width` (int): 图像的宽高。
+ `image_id` (str or int): 标识这幅图像的唯一ID。很多评估器需要识别这幅图像，但是数据集可能被用于不同用途。
+ `annotations` (list[dict]): __实例检测/分割__ 或 __关键点检测__ 任务需要。一个字典对应于一个实例的标注，它可能会包含以下键：
  + `bbox` (list[float], required): 代表实例边界框的列表，由4个数字组成。
  + `bbox_mode` (int, required): 边界框格式。它必须是
    [structures.BoxMode](../modules/structures.html#detectron2.structures.BoxMode) 的成员变量。
    当前支持： `BoxMode.XYXY_ABS`， `BoxMode.XYWH_ABS`。
  + `category_id` (int, required): 代表类别标签的整数，其数值范围是 [0,类别数量-1]。
  数值大小为类别数量的数被保留用来表示"背景"类别（如果适用的话）。
  + `segmentation` (list[list[float]] or dict): 实例的分割掩码。
    + 若数据结构是`list[list[float]]`， 则代表是多边形列表，每个多边形代表一个目标的组件。 每个 `list[float]` 代表一个多边形，形式为 `[x1, y1, ..., xn, yn]` （n≥3）。 Xs 和 Ys 是以像素为单位的绝对坐标。
    + 若数据结构是 `dict`， 则代表像素掩码保存格式是 COCO 压缩的 RLE 格式。
      字典应包含有键 "size" 和 "counts"。 您可以使用 `pycocotools.mask.encode(np.asarray(mask, order="F"))` 将一个 uint8 的分割掩码转换成用0和1表示的字典。
      若在默认数据加载器下使用这种格式，`cfg.INPUT.MASK_FORMAT` 须设置为 `bitmask`。
  + `keypoints` (list[float]): 格式为 [x1, y1, v1,..., xn, yn, vn]。
    v[i] 表示点的[可见性](http://cocodataset.org/#format-data)。
    `n` 须等于关键点类别的数量。
    Xs 和 Ys 是绝对坐标， 范围在 [0, W 或 H]。

    （注意 COCO 格式中关键点坐标的范围是 [0， W-1 or H-1]，
    这与我们的标准格式不同。 Detectron2 在 COCO 关键点坐标上加了0。5，
    将其从离散的像素索引转换为浮点形式的坐标。）
  + `iscrowd`: 0 （默认） 或 1。 表示该实例是否被标记为 COCO 中的 "crowd
    region"。 如果您不清楚此字段的含义，就不要包含此字段。

  若 `annotations` 是空列表， 表示该图像被标记为没有对象。
  默认情况下将从训练中删除，
  但可以通过设置 `DATALOADER.FILTER_EMPTY_ANNOTATIONS` 将其包含在训练中。

+ `sem_seg_file_name` (str):
  语义分割标注文件的完整路径。
  该文件应该是像素值为整数的灰度图像。
+ `pan_seg_file_name` (str):
  全景分割标注文件的完整路径。
  该标注文件的像素值是使用 [panopticapi.utils.id2rgb](https://github.com/cocodataset/panopticapi/) 函数编码的整数 id，
  id 由 `segments_info` 定义。
  若 id 没有被 `segments_info` 定义，该像素将被视为未标注，
  在训练和评估中通常会被忽略。
+ `segments_info` (list[dict]): 定义了全景分割标注中每个 id 的含义。
  每个字典包含以下键：
  + `id` (int): 标注文件中的像素值。
  + `category_id` (int): 范围在 [0, 类别数-1] 代表类别标签的整数。
  + `iscrowd`： 0 （默认） 或 1。 代表该实例是否是 COCO 中的 "crowd region"。

```eval_rst
.. note：：

   PanopticFPN 模型不使用此处定义的全景分割格式，
   而是使用实例分割和语义分割数据格式的组合。
   有关 COCO 的说明，请参阅 :doc:`builtin_datasets`。

```

Fast R-CNN（带有预计算的待选区） 如今已很少被使用。
要训练 Fast R-CNN，需要以下额外键：

+ `proposal_boxes` (array): 形状为 (K, 4) 的 2D numpy 数组，代表此图片有 K 个预计算的待选区。
+ `proposal_objectness_logits` (array): 形状为 (K, ) 的 numpy 数组， 代表 `proposal_boxes` 中待选区的目标对数机率。
+ `proposal_bbox_mode` (int): 预计算待选区域的格式。
 须是 [structures.BoxMode](../modules/structures.html#detectron2.structures.BoxMode) 的成员变量。
 默认是 `BoxMode.XYXY_ABS`。


#### 为新任务自定义数据集字典

数据集函数中返回的 `list[dict]` 中的字典可以包含 __任意自定义数据__ 。
这对于一些需要额外信息，而这些信息标准数据集字典又未涵盖的新任务非常游泳。
在这种情况下，您需要确保下游代码可以正确处理您的数据。
通常这需要为数据加载器编写新的 `mapper` （请参见[使用自定义数据加载器](./data_loading.md)）。

在设计自定义格式时，请注意所有字典都存储在内存中（有时是序列化的，并且具有多个副本）。
为了节省内存，每个字典都包含有关每个样本 __少量__ 但是足够的信息，比如文件名和标注。
在数据加载器中才加载完整的样本。

对于整个数据集间共享的属性，请使用 `元数据` （见下文）。
为了避免额外内存消耗，请勿将这类信息保存在每个样本中。

### 数据集的"元数据"

每个数据集都和一些元数据相关联，可以通过访问
`MetadataCatalog.get(dataset_name).some_metadata`获得。
元数据是一种键值映射，其中包含在整个数据集之间共享的信息，
通常用于解释数据集中的内容，例如，类的名称，类的颜色，文件的根目录等。
此信息对于数据增强，评估，可视化，日志记录等非常有用。
元数据的结构取决于相应下游代码需要的内容。

若通过 `DatasetCatalog.register` 注册新数据集，
您可能还希望通过 `MetadataCatalog.get(dataset_name).some_key = some_value` 添加其想要的元数据，
用以支持其他需要元数据的功能。
您可以像以下这样进行操作（以元数据键为 "thing_classes" 为例）：

```python
from detectron2.data import MetadataCatalog
MetadataCatalog.get("my_dataset").thing_classes = ["person", "dog"]
```

以下是 detectron2 中内置功能所使用的元数据键的列表。
若您添加的自有数据集缺少这些元数据，一些功能可能不可用：

* `thing_classes` (list[str]): 由实例检测/分割任务的所有实例所使用。代表每个实例/事物（thing）类别的名称列表。
  若加载 COCO 格式数据集， `load_coco_json`函数将会自动设置此字段。

* `thing_colors` (list[tuple(r, g, b)]): 每个事物（thing）类别的预定义颜色（范围在 [0, 255]）。
  用作可视化。若未给出，将使用随机颜色。

* `stuff_classes` (list[str]): 由语义分割或全景分割任务所使用。代表每个填充（stuff）类别的名称列表。

* `stuff_colors` (list[tuple(r, g, b)]): 每个填充（stuff）类别的预定义颜色（范围在 [0, 255]）。
  用作可视化。若未给出，将使用随机颜色。

* `ignore_label` (int): 由语义分割或全景分割任务所使用。在评估中将忽略具有此类标签的答案标注。
  通常这些都是"未标注"的像素。

* `keypoint_names` (list[str]): 由关键点检测任务所使用。代表每个关键点的名称列表。

* `keypoint_flip_map` (list[tuple[str]]): 由关键点检测任务所使用。是名称对的列表，
  其中每对是图像在进行水平翻转增强时应翻转的两个关键点。

* `keypoint_connection_rules`: list[tuple(str, str, (r, g, b))]。 指明了每对相互链接的关键点在可视化时的颜色（范围在 [0, 255]）。

以下是一些特定数据集（例如 COCO）在评估时需要指定的额外元数据。

* `thing_dataset_id_to_contiguous_id` (dict[int->int]): 由所有采用 COCO 格式的实例检测/分割任务使用。
  将数据集中实例类别的 id 映射成范围在 [0,#class) 的连续 id。
  通常由 `load_coco_json` 函数自动设置。

* `stuff_dataset_id_to_contiguous_id` (dict[int->int]): 在生成用于语义/全景分割任务的预测 json 文件中使用。
  将数据集中的语义分割类别 id 映射到范围 [0， 类别总数)的连续id。仅在评估时使用

* `json_file`: COCO 标注的json文件。使用 COCO 方式评估 COCO 格式数据集使用。

* `panoptic_root`, `panoptic_json`: 由 COCO 格式的全景分割评估使用。

* `evaluator_type`: 主训练脚本用来选择评估器时使用。请勿在新的训练脚本中使用。
   您只需直接在您的训练脚本中直接为您的数据集提供 [DatasetEvaluator](../modules/evaluation.html#detectron2.evaluation.DatasetEvaluator)

```eval_rst
.. note：：

   在识别中，有时我们会使用"事物（ thing ）"来指代实例级任务，
   使用"填充（ stuff ）"来指代语义分割级任务。
   在全景分割任务中，两者皆有使用。
   对于两者概念，参见
   `On Seeing Stuff: The Perception of Materials by Humans and Machines
   <http://persci.mit.edu/pub_pdfs/adelson_spie_01.pdf>`_。
```

```eval_rst
.. note：：

    译者注：
    在分割任务中， 事物（thing）常被代指可以单独识别区分的离散对象，比如人，车，动物等。这些对象在图像中可以实例化，可以被单独框选，每个实例由自己的边界框或者分割掩码。
    填充（stuff）常用来指代不进行实例区分，没有明显边界的连续区域，比如背景中的道路，天空，地面等等。

```

### 注册 COCO 格式数据集

若您的实例级（检测，分割，关键点）数据集是用一个 COCO 格式的 json 文件标注的。
则通过以下方式可以轻松注册数据集和其关联的元数据：
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
```

若您的数据集使用的是 COCO 格式，且需要进一步处理，或每个实例由一些额外的自定义标注，
[load_coco_json](../modules/data.html#detectron2.data.datasets.load_coco_json) 函数可能会很有用。

### 更新新数据集配置

一旦您注册了数据集，您可以在 `cfg.DATASETS.{TRAIN，TEST}` 配置项中使用数据集的名字（比如，上面示例中的 "my_dataset"）。
您可能希望更改其他配置以在新数据集上进行训练或评估。

* `MODEL.ROI_HEADS.NUM_CLASSES` 和 `MODEL.RETINANET.NUM_CLASSES` 分别是 R-CNN 和 RetinaNet 模型的事物（thing）类别数。

* `MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS` 设置 R-CNN 关键点模型的关键点数量。
  您还需要 `TEST.KEYPOINT_OKS_SIGMAS` 设置为 [Keypoint OKS](http://cocodataset.org/#keypoints-eval)
  以用于评估。

* `MODEL.SEM_SEG_HEAD.NUM_CLASSES` 是语义 FPN 或全景 FPN 中填充（stuff）类别数。

* `TEST.DETECTIONS_PER_IMAGE` 控制检测的最大目标数。若单张测试图像中包含超过100个目标，需要设置一个更大的值。


* 若要训练 Fast R-CNN（带有预计算的待选区）， `DATASETS.PROPOSAL_FILES_{TRAIN，TEST}`
  需要匹配数据集。 待选区的格式参见[这里](../modules/data.html#detectron2.data.load_proposals_into_dataset)。

新模型
（比如 [TensorMask](../../../projects/TensorMask)，
[PointRend](../../../projects/PointRend)）
通常具有自己的类似配置，也需要更改。

```eval_rst
.. tip：：

   更改类数后，预训练模型中的某些层将变得不兼容，因此无法加载到新模型中。这是意料之中的，加载此类预训练模型将生成有关此类层的警告。

```
