## 安装

### 环境要求
- 装有 Python ≥ 3.7 的 Linux 或 macOS
- 装有 PyTorch ≥ 1.8 和对应版本的 [torchvision](https://github.com/pytorch/vision/),
  通过在 [pytorch.org](https://pytorch.org) 一起安装它们可以确保版本一致
- 若需要演示和可视化,还需要安装 OpenCV


### 源码构建 Detectron2

gcc & g++ ≥ 5.4 are required.  is optional but recommended for faster build.
gcc & g++ ≥ 5.4 是必需的.[ninja](https://ninja-build.org/) 可选,但建议安装,可以加快构建.
满足这些条件后,运行
```shell
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# On macOS, you may need to prepend the above commands with a few environment variables:
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...
```

若要从本地克隆仓库中__重新构建__ detectron2,请先执行 `rm -rf build/ **/*.so` 来清理之前的构建.
在重新安装 PyTorch 之后, detectron2 也需要重新构建.

### 安装预构建的 Detectron2 (仅 Linux)

根据此表可安装 [v0.6 (Oct 2021)](https://github.com/facebookresearch/detectron2/releases):

<table class="docutils"><tbody><th width="80"> CUDA </th><th valign="bottom" align="left" width="100">torch 1.10</th><th valign="bottom" align="left" width="100">torch 1.9</th><th valign="bottom" align="left" width="100">torch 1.8</th> <tr><td align="left">11.3</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
</code></pre> </details> </td> <td align="left"> </td> <td align="left"> </td> </tr> <tr><td align="left">11.1</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
</code></pre> </details> </td> </tr> <tr><td align="left">10.2</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html
</code></pre> </details> </td> </tr> <tr><td align="left">10.1</td><td align="left"> </td> <td align="left"> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
</code></pre> </details> </td> </tr> <tr><td align="left">cpu</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
</code></pre> </details> </td> </tr></tbody></table>


注意:
1. 预构建的包必须与相应版本的 CUDA 和 PyTorch 的官方包一起使用.否则,请从源代码构建 detectron2.
2. 每隔几个月就会发布一次新软件包.因此,软件包可能不包含主分支中的最新功能,并且可能与使用 detectron2 的研究项目的主分支(例如[项目中这些功能](projects))不兼容.

### 常见安装问题

单击每个问题以获取其解决方案:

<details>
<summary>

有类似 "TH..","at::Tensor...","torch..." 等符号未定义.

</summary>
<br/>

这类问题通常是因为 detectron2 或 torchvision 与正在运行的 PyTorch 版本不匹配导致的.

若是预构建的 torchvision 报错,请卸载 torchvision 和 pytorch,并根据 [pytorch.org](http://pytorch.org) 来重新安装它们,以确保两者版本匹配.

若是预构建的 detectron2 报错,请检查 [release notes](https://github.com/facebookresearch/detectron2/releases),卸载当前 detectron2 并重新安装正确的和 pytorch 版本匹配的预构建 detectron2.

若是手动构建的 detectron2 或 torchvision 报错,请删除手动构建文件(`build/`,`**/*.so`)并重新构建,以便可以获取您当前环境中存在的 pytorch 版本.

若上述方案均无法解决问题,请提供可以复现问题的环境(比如 dockerfile).

</details>

<details>
<summary>

使用 detectron2 提示缺少 torch 的动态链接库或是发生 segmentation fault.

</summary>
<br/>

这类问题通常时因为 detectron2 或 torchvision 和当前正在运行的 PyTorch 版本不匹配导致的.解决方法参见上一个问题.

</details>

<details>
<summary>

未定义或者未找到 C++ 符号(比如 "GLIBCXX..").

</summary>
<br/>

这通常是因为库使用了较新的 C++ 编译器编译,但运行环境下的 C++ 运行库是旧的.

这类问题在较旧的 anaconda 上易出现,运行 `conda update libgcc` 来升级 C++ 运行库可能会有所帮助.

解决方案的根本在于要避免 C++ 编译器不匹配问题,要么使用较旧的 C++ 编译器,要么使用合适的 C++ 运行库.
若要指定 C++ 运行库,可以使用环境变量 `LD_PRELOAD=/path/to/libstdc++.so`.

</details>

<details>
<summary>

"nvcc not found" 或 "Not compiled with GPU support" 或 "Detectron2 CUDA Compiler: not available".

</summary>
<br/>

构建 detectron2 时未找到 CUDA.你应该确保在你构建 detectron2 时运行  

```shell
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```

打印结果为 `(True, a directory with cuda)` .

大部分模型都可以在无 GPU 支持的情况下推理(但不能训练).若要使用 CPU,请在配置中设置 `MODEL.DEVICE='cpu'`.

</details>

<details>
<summary>

"invalid device function" 或 "no kernel image is available for execution".

</summary>
<br/>

导致该问题的两个可能原因:

* 构建 detectron2 时的 CUDA 和运行时的 CUDA 版本不一致.

  要确认是否时这类情况,请使用 `python -m detectron2.utils.collect_env` 来找出不一致的 CUDA 版本.理想情况下, 
  这条命令输出中的 "Detectron2 CUDA Compiler", "CUDA_HOME", "PyTorch built with - CUDA" 中包含的 cuda 库版本应该一致.

  当它们不一致时,您需要安装不同的 PyTorch 版本(或自己构建) 以匹配本地 CUDA 安装，或安装其他版本的 CUDA 以匹配 PyTorch.

* PyTorch/torchvision/Detectron2 不是以正确的 GPU SM 架构(又名计算能力)构建的。

  指令`python -m detectron2.utils.collect_env` 中的 "architecture flags" 显示有 PyTorch/detectron2/torchvision 的架构.
  这些架构中必须包含有您的 GPU 架构, GPU 架构可参见 [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).

  若您使用的是预构建的 PyTorch/detectron2/torchvision,它们通常已经支持了最流行的一些 GPU.
  若不支持,您需要自行从源码构建它们.

  从源代码构建 detectron2/torchvision 时,它们会检测 GPU 设备并仅针对设备进行构建.
  这意味着编译后的代码可能无法在其他 GPU 设备上运行.
  要以正确的体系结构重新编译它们，请删除所有已安装/编译的文件,
  并使用正确设置 `TORCH_CUDA_ARCH_LIST` 环境变量并重新生成它们.
  例如, `export TORCH_CUDA_ARCH_LIST="6.0;7.0"` 使其同时针对 P100 和 V100 进行编译.

</details>

<details>
<summary>
Undefined CUDA symbols; Cannot open libcudart.so
</summary>
<br/>
The version of NVCC you use to build detectron2 or torchvision does
not match the version of CUDA you are running with.
This often happens when using anaconda's CUDA runtime.

Use `python -m detectron2.utils.collect_env` to find out inconsistent CUDA versions.
In the output of this command, you should expect "Detectron2 CUDA Compiler", "CUDA_HOME", "PyTorch built with - CUDA"
to contain cuda libraries of the same version.

When they are inconsistent,
you need to either install a different build of PyTorch (or build by yourself)
to match your local CUDA installation, or install a different version of CUDA to match PyTorch.
</details>


<details>
<summary>
C++ compilation errors from NVCC / NVRTC, or "Unsupported gpu architecture"
</summary>
<br/>
A few possibilities:

1. Local CUDA/NVCC version has to match the CUDA version of your PyTorch. Both can be found in `python collect_env.py`
   (download from [here](./detectron2/utils/collect_env.py)).
   When they are inconsistent, you need to either install a different build of PyTorch (or build by yourself)
   to match your local CUDA installation, or install a different version of CUDA to match PyTorch.

2. Local CUDA/NVCC version shall support the SM architecture (a.k.a. compute capability) of your GPU.
   The capability of your GPU can be found at [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).
   The capability supported by NVCC is listed at [here](https://gist.github.com/ax3l/9489132).
   If your NVCC version is too old, this can be workaround by setting environment variable
   `TORCH_CUDA_ARCH_LIST` to a lower, supported capability.

3. The combination of NVCC and GCC you use is incompatible. You need to change one of their versions.
   See [here](https://gist.github.com/ax3l/9489132) for some valid combinations.
   Notably, CUDA<=10.1.105 doesn't support GCC>7.3.

   The CUDA/GCC version used by PyTorch can be found by `print(torch.__config__.show())`.

</details>


<details>
<summary>
"ImportError: cannot import name '_C'".
</summary>
<br/>
Please build and install detectron2 following the instructions above.

Or, if you are running code from detectron2's root directory, `cd` to a different one.
Otherwise you may not import the code that you installed.
</details>


<details>
<summary>
Any issue on windows.
</summary>
<br/>

Detectron2 is continuously built on windows with [CircleCI](https://app.circleci.com/pipelines/github/facebookresearch/detectron2?branch=main).
However we do not provide official support for it.
PRs that improves code compatibility on windows are welcome.
</details>

<details>
<summary>
ONNX conversion segfault after some "TraceWarning".
</summary>
<br/>
The ONNX package is compiled with a too old compiler.

Please build and install ONNX from its source code using a compiler
whose version is closer to what's used by PyTorch (available in `torch.__config__.show()`).
</details>


<details>
<summary>
"library not found for -lstdc++" on older version of MacOS
</summary>
<br/>

See [this stackoverflow answer](https://stackoverflow.com/questions/56083725/macos-build-issues-lstdc-not-found-while-building-python-package).

</details>


### Installation inside specific environments:

* __Colab__: see our [Colab Tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
  which has step-by-step instructions.

* __Docker__: The official [Dockerfile](docker) installs detectron2 with a few simple commands.
