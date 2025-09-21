# 深度学习-CV

## `torchvision.transforms`

### `transforms.ToTensor()`

`transforms.ToTensor()` 是 PyTorch 中一个**将图像数据转换为张量格式**的转换操作。

- 数据类型变化：转换前`PIL.Image` 或 `numpy.ndarray`，转换后`torch.Tensor`
- 数值范围缩放：转换前 像素值范围 `[0, 255] (uint8)`，转换后 数值范围 `[0.0, 1.0] (float32)`
- 维度顺序调整：转换前 `(H, W, C) - 高度、宽度、通道`，转换后 `(C, H, W) - 通道、高度、宽度`

### `transforms.Resize()`

`Resize` 是一个用于调整图像（和对应的目标，如边界框，如果适用）尺寸的变换（Transform）。它的主要目的是将输入图像的大小统一调整为指定的尺寸。

- `size` (必需参数): 期望的输出尺寸。有几种提供方式：
  - **`(height, width)`**: 一个整数元组，直接指定调整后的高度和宽度。
    - `transforms.Resize((256, 128))` → 将图像调整为高 256 像素，宽 128 像素。
  - **`size`**: 一个整数。图像的较短边将被缩放至这个尺寸，较长边按比例缩放，以保持图像的原始宽高比。
    - `transforms.Resize(256)` → 如果原图是 (500, 300) [H x W]，那么较短边是 300。调整后，新尺寸为 `(256 * 500/300, 256)`，即 ≈ (427, 256)。这种方式非常常用，因为它避免了图像失真。
- `interpolation` (可选参数): 插值方法，用于决定如何计算新的像素值。默认为 `InterpolationMode.BILINEAR`。
  - `InterpolationMode.BILINEAR`: 双线性插值。质量和速度的较好平衡，最常用。
  - `InterpolationMode.NEAREST`: 最近邻插值。速度快，但可能产生锯齿状边缘。
  - `InterpolationMode.BICUBIC`: 双三次插值。质量更高，但速度更慢。

### `transforms.Compose`

`Compose` 不是一个具体的图像变换操作，而是一个**容器**或**流水线构建器**。它将多个变换操作（Transform）组合成一个单一的、可调用的变换流水线，操作后可以按顺序定义一系列预处理步骤，然后一次性应用到图像上，代码会变得非常清晰和模块化。

向 `Compose` 的构造函数传递一个由多个 `transforms` 对象组成的列表。当调用这个 `Compose` 对象时，它会按顺序依次对输入图像应用列表中的每一个变换。

#### 示例代码

假设一个典型的预处理流程是：

1. 随机裁剪图像
2. 调整大小到 224x224
3. 转换为 PyTorch 张量
4. 对张量进行标准化（归一化）

没有 `Compose` 时，代码会很冗长：

```python
image = Image.open("image.jpg").convert('RGB')
image = transforms.RandomCrop(32)(image)      # 步骤1
image = transforms.Resize((224, 224))(image)  # 步骤2
tensor = transforms.ToTensor()(image)         # 步骤3
tensor = transforms.Normalize(...)(tensor)    # 步骤4
```

使用 `Compose` 后，代码变得非常简洁和易管理：

```python
from torchvision import transforms

# 定义预处理流水线
preprocess = transforms.Compose([
    transforms.RandomCrop(32),        # 步骤1
    transforms.Resize((224, 224)),    # 步骤2
    transforms.ToTensor(),            # 步骤3
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])  # 步骤4
])

# 应用整个流水线
image = Image.open("path_to_your_image.jpg").convert('RGB')
processed_tensor = preprocess(image) 
# processed_tensor 现在是一个可以直接送入神经网络的标准化张量

print(processed_tensor.shape) 
# 输出: torch.Size([3, 224, 224])
```

#### 重要注意事项

- **顺序至关重要**：变换的顺序会影响最终结果。例如，`ToTensor()` 必须在对张量的操作（如 `Normalize`）之前，因为 `Normalize` 需要张量作为输入，而不能是 PIL Image。

- **训练和验证的流水线通常不同**：训练时你可能需要数据增强（如随机裁剪、水平翻转），而验证/测试时你通常只需要简单的调整大小和标准化。因此，你通常会定义两个不同的 `Compose` 流水线。

  ```python
  # 训练数据变换 (包含数据增强)
  train_transforms = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(...)
  ])
  
  # 验证/测试数据变换 (不包含数据增强)
  val_transforms = transforms.Compose([
      transforms.Resize(256),           # 保持比例缩放
      transforms.CenterCrop(224),       # 从中心裁剪
      transforms.ToTensor(),
      transforms.Normalize(...)
  ])
  ```

  