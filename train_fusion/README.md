# LLaVA-OCR 特征融合训练框架

本框架实现了将OCR功能集成到LLaVA模型中的特征融合方案，旨在增强模型对图像中文本的理解能力。

## 核心思想

本方案的核心思想是：**将OCR提取的精确文本位置与内容，与LLaVA视觉编码器提取的全局视觉语义进行融合**。具体流程如下：

1. **特征并行提取**：
   - **OCR路径**：使用PaddleOCR处理图像，提取文本内容和位置信息
   - **LLaVA视觉编码器路径**：使用CLIP-ViT提取图像的全局视觉特征

2. **特征对齐与映射**：
   - 将OCR生成的文本描述转换为文本特征向量
   - 将LLaVA的视觉特征通过投影层映射到语言模型空间

3. **特征融合**：将上述两种特征进行拼接，形成综合特征表示

4. **语言模型推理**：将融合特征输入到LLaVA的语言模型中生成回答

## 目录结构

```
train_fusion/
├── config.json           # 配置文件，设置模型和训练参数
├── feature_extraction.py # 特征提取模块，提取LLaVA特征和OCR特征
├── feature_fusion.py     # 特征融合模块，实现特征拼接和投影
├── llava_ocr_model.py    # LLaVA-OCR融合模型实现
├── train_script.py       # 训练脚本，包含数据集、训练器等
├── main.py               # 主入口文件，启动训练
└── README.md             # 说明文档
```

## 训练策略

根据用户要求，我们采用以下训练策略（针对单V100 GPU环境优化）：

- **冻结OCR模型**：不训练，使用预训练的PaddleOCR模型
- **冻结视觉编码器**：不训练CLIP-ViT，避免遗忘通用视觉知识
- **训练投影层**：训练MLP投影层，实现视觉-语言特征对齐
- **LoRA微调语言模型**：使用LoRA技术对LLaVA的大语言模型进行参数高效微调

## 配置说明

配置文件 `config.json` 包含以下主要部分：

- **model_config**：模型配置，包括LLaVA和OCR模型的路径
- **training_config**：训练配置，包括批量大小、学习率、训练轮数等
- **lora_config**：LoRA配置，包括是否启用LoRA、LoRA秩等参数
- **data_config**：数据配置，包括训练和验证数据集的路径

## 使用方法

### 前提条件

1. 已安装必要的依赖：
   ```bash
   pip install torch transformers paddleocr peft pandas pillow tqdm
   ```

2. 已下载LLaVA模型和PaddleOCR模型，并配置到`config.json`中

3. 已准备好训练数据集，格式为parquet文件

### 启动训练

```bash
cd /root/autodl-tmp/codes/Vqa_ocr/train_fusion
python main.py
```

也可以指定自定义配置文件：

```bash
python main.py /path/to/your/config.json
```

### 输出结果

训练过程中的损失值会实时显示在控制台，训练好的模型权重将保存到`/root/autodl-tmp/weight`目录下。

## 性能优化

本方案针对单V100 GPU环境进行了优化：

1. 使用混合精度训练（FP16）减少显存占用
2. 实现梯度累积，模拟更大的批量大小
3. 使用LoRA技术大幅减少可训练参数量
4. 冻结视觉编码器和OCR模型，只训练必要的模块

## 注意事项

1. 确保数据集格式正确，包含image、question和answer字段
2. 根据实际GPU显存大小调整批量大小和梯度累积步数
3. 训练过程中可能需要调整学习率以获得更好的性能
4. 如遇到OCR初始化失败，请检查PaddleOCR是否正确安装

## 技术特点

1. **端到端训练**：整个融合过程在一个端到端的框架中完成
2. **参数高效**：采用LoRA技术，在保证效果的同时减少计算成本
3. **灵活配置**：通过配置文件可以灵活调整各种参数
4. **位置感知**：OCR提取的文本位置信息被整合到特征表示中

---

通过本框架，您可以训练一个同时具备强大视觉理解能力和精确文本识别能力的多模态模型，为各种视觉问答任务提供更准确的回答。