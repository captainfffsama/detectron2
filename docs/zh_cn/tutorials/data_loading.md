
# 数据加载器

数据加载器是为模型提供数据的组件。
数据加载器通常（但不一定）从[数据集](./datasets.md)中获取原始信息，
并将其处理为模型所需的格式。

## 现有数据加载器的工作原理

Detectron2 包含了一个内置的数据加载流程。
了解其工作原理将有助于您编写一个自定义的数据加载流程。

Detectron2 提供了 [build_detection_{train，test}_loader](../modules/data.html#detectron2.data.build_detection_train_loader) 两个函数，
用于从给定配置中创建默认数据加载器。
以下是 `build_detection_{train,test}_loader` 的工作原理：

1. 它会读取注册数据集的名称（如，"coco_2017_train"），并使用 `list[dict]` 来表示加载到的轻量格式的数据集项。
   但此时这些轻量格式的数据集项并不能直接供模型使用（比如，图像文件没有被加载到内存中，没有对样本应用随机增强等）。
   有关数据集格式和注册方法，可参见文档
   [数据集](./datasets.md)。
2. 该 list 中每个 dict 都是由映射器 （"mapper"） 映射而来的：
   * 通过指定 `build_detection_{train,test}_loader` 中 "mapper" 参数，可以传入自定义的映射器。默认映射器是
        [DatasetMapper](../modules/data.html#detectron2.data.DatasetMapper)。
   * 映射器的输出可以是任意格式，只要该格式可以被下游数据加载器的消费者（通常是模型）接受即可。
     默认映射器的输出，在经过批处理后，将遵循默认模型的输入格式，默认模型的输入格式参见文档
     [使用模型](./models.html#model-input-format)。
   * 映射器的作用是将数据集项的轻量格式转换成供模型消费的格式（包括读取图像，执行随机数据增强，转换为 torch 张量等）。
     若您需要对数据进行自定义转换，通常需要自定义一个映射器。
3. 映射器的输出将被批处理（简单地放入一个列表）。
4. 此批处理的数据是数据加载器的输出。通常，它也是
   `model.forward()` 的输入。


## 编写自定义数据加载器

`build_detection_{train，test}_loader(mapper=)` 配合不同的 "映射器" 可以适用于大多数自定义加载的情况。
例如，若想将所有图像调整为固定尺寸来训练，可以使用如下操作：

```python
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper
dataloader = build_detection_train_loader(cfg,
   mapper=DatasetMapper(cfg, is_train=True, augmentations=[
      T.Resize((800, 800))
   ]))
# use this dataloader instead of the default
```

若默认的映射器 [DatasetMapper](../modules/data.html#detectron2.data.DatasetMapper) 的参数不能满足需求，
您可以编写并使用一个自定义映射器，例如：

```python
from detectron2.data import detection_utils as utils
 # Show how to implement a minimal mapper, similar to the default DatasetMapper
def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # can use other ways to read image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # See "Data Augmentation" tutorial for details usage
    auginput = T.AugInput(image)
    transform = T.Resize((800, 800))(auginput)
    image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]
    return {
       # create the format that the model expects
       "image": image,
       "instances": utils.annotations_to_instances(annos, image.shape[1:])
    }
dataloader = build_detection_train_loader(cfg, mapper=mapper)
```

若您想要更改的不仅仅是映射器（比如，还想实现不同的采样或者批处理逻辑），
`build_detection_train_loader` 将不起作用，您还需要编写一个不同的数据加载器。
数据加载器只是一个简单的 python 迭代器，用来产生模型可以接受的[格式](./models.md)。
您可以使用任何您喜欢的工具来实现它。

无论如何实现，都推荐参看 [detectron2.data 的 API 文档](../modules/data) 来了解有关这些函数 API 的信息。

## 使用自定义数据加载器

若您使用 [DefaultTrainer](../modules/engine.html#detectron2.engine.defaults.DefaultTrainer)，
您可以通过覆盖其 `build_{train，test}_loader` 方法来使用您自己的数据加载器。
相关示例，可以参见 [deeplab dataloader](../../../projects/DeepLab/train_net.py)。

若您自己编写了训练循环，则可以轻松插入数据加载器。
