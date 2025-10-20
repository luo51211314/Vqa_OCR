import torch
import torch.nn as nn
import json
import sys
import os
# 设置环境变量，确保使用本地缓存的CLIP权重
# 强制离线模式，不进行任何网络请求
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface/hub"
# 将上级目录和llava目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'llava'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_loader'))
# 导入模型加载相关模块
from models.llava.llava.model.builder import load_pretrained_model
from models.llava.llava.mm_utils import get_model_name_from_path
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from feature_fusion import FusedVisionProjector

class LLaVAOCRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.llava_model = None
        self.tokenizer = None
        self.fused_projector = None
        self.initialized = False
        
    def initialize(self):
        """初始化模型、分词器和融合投影器"""
        if not self.initialized:
            # 配置量化参数（如果需要）
            load_4bit = (self.config['training_config'].get('bits', 16) == 4)
            
            # 按照loader_llava的方式加载LLaVA模型，使用本地llava权重和分词器
            # 指定本地llava模型路径
            model_path = self.config['model_config'].get('llava_model_path', '/root/autodl-tmp/model/llava_hug')
            model_name = get_model_name_from_path(model_path)
            
            # 配置加载参数 - 使用FP32权重
            kwargs = {
                "device_map": "auto",
                "dtype": torch.float32
            }
            
            if load_4bit:
                kwargs['load_in_4bit'] = True
                kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                )
            
            # 加载模型，使用本地路径
            self.tokenizer, self.llava_model, self.image_processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                **kwargs
            )
            
            # 保存model_config以便在图像处理中使用
            self.model_config = self.llava_model.config
            
            # 重新设置分词器参数，确保符合训练配置
            self.tokenizer.model_max_length = self.config['training_config']['max_length']
            self.tokenizer.padding_side = "right"
            
            # 初始化融合投影器
            self.fused_projector = FusedVisionProjector(self.config)
            self.fused_projector.initialize(self.llava_model, self.tokenizer)
            
            # 配置LoRA（如果需要）
            if self.config['lora_config']['lora_enable']:
                self._configure_lora()
            
            # 配置训练参数
            self._configure_training_parameters()
            
            self.initialized = True
        
        return self
        
    def _configure_lora(self):
        """配置LoRA参数高效微调"""
        lora_config = LoraConfig(
            r=self.config['lora_config']['lora_r'],
            lora_alpha=self.config['lora_config']['lora_alpha'],
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.config['lora_config']['lora_dropout'],
            bias=self.config['lora_config']['lora_bias'],
        )
        
        self.llava_model = get_peft_model(self.llava_model, lora_config)
        self.llava_model.print_trainable_parameters()
        
    def _configure_training_parameters(self):
        """配置训练参数，冻结不需要训练的模块"""
        # 冻结视觉编码器
        vision_tower = self.llava_model.get_vision_tower()
        if vision_tower is not None:
            for param in vision_tower.parameters():
                param.requires_grad = False
        
        # 确保投影层可训练
        if hasattr(self.llava_model.get_model(), 'mm_projector'):
            for param in self.llava_model.get_model().mm_projector.parameters():
                param.requires_grad = True
        
        # 冻结LLM的部分参数（如果没有使用LoRA）
        if not self.config['lora_config']['lora_enable']:
            # 这里可以根据需要冻结LLM的某些层
            pass
        
        # 确保fused_projector的ocr_text_projector可训练
        if hasattr(self.fused_projector.feature_fusion, 'ocr_text_projector'):
            for param in self.fused_projector.feature_fusion.ocr_text_projector.parameters():
                param.requires_grad = True
        
    def forward(self, images, input_ids, attention_mask=None, labels=None):
        """前向传播"""
        if not self.initialized:
            self.initialize()
        
        # 使用融合投影器获取投影后的特征
        fusion_result = self.fused_projector(images)
        projected_features = fusion_result['projected_features']
        
        # 准备输入
        # 检查images是列表还是张量，获取正确的batch_size
        if isinstance(images, list):
            batch_size = len(images)
        else:
            batch_size = images.shape[0]
        
        # 这里需要重构LLaVA的前向传播逻辑，使用我们的融合特征
        # 我们将替换原始的encode_images和prepare_inputs_labels_for_multimodal逻辑
        
        # 为每个图像添加一个image token
        image_token_index = self.tokenizer.convert_tokens_to_ids(["<image>"])[0]
        
        # 查找输入中的image token位置
        image_positions = []
        for i in range(batch_size):
            positions = (input_ids[i] == image_token_index).nonzero().squeeze().tolist()
            if isinstance(positions, int):
                positions = [positions]
            image_positions.append(positions)
        
        # 准备新的输入嵌入
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        
        for i in range(batch_size):
            # 获取当前样本的输入嵌入
            cur_input_embeds = self.llava_model.get_model().embed_tokens(input_ids[i])
            
            # 插入融合特征
            if len(image_positions[i]) > 0:
                # 假设只有一个image token
                pos = image_positions[i][0]
                
                # 分割嵌入向量
                before_image = cur_input_embeds[:pos]
                after_image = cur_input_embeds[pos+1:]
                
                try:
                    # 获取当前样本的投影特征
                    proj_feature = projected_features[i]
                    
                    # 确保投影特征维度正确
                    # 关键修复：确保投影特征的维度与LLaVA特征一致
                    if proj_feature.dim() == 3:
                        # 如果是[1, seq_len, hidden_size]，则去掉第一维
                        if proj_feature.shape[0] == 1:
                            proj_feature = proj_feature.squeeze(0)
                    
                    # 确保是2D张量 [seq_len, hidden_size]
                    if proj_feature.dim() == 1:
                        proj_feature = proj_feature.unsqueeze(0)
                    
                    # 确保所有张量在同一设备上
                    proj_feature = proj_feature.to(before_image.device)
                    
                    # 确保before_image和after_image也是正确的维度
                    if before_image.dim() == 1:
                        before_image = before_image.unsqueeze(0)
                    if after_image.dim() == 1:
                        after_image = after_image.unsqueeze(0)
                    
                    # 执行拼接
                    fused_embeds = torch.cat([before_image, proj_feature, after_image], dim=0)
                    
                    new_input_embeds.append(fused_embeds)
                    
                    # 处理标签
                    if labels is not None:
                        before_image_labels = labels[i][:pos]
                        after_image_labels = labels[i][pos+1:]
                        # 为image features添加ignore index
                        image_labels = torch.full((proj_feature.shape[0],), 
                                                 self.config.get('ignore_index', -100), 
                                                 device=labels.device, dtype=labels.dtype)
                        
                        new_label = torch.cat([before_image_labels, image_labels, after_image_labels], dim=0)
                        new_labels.append(new_label)
                except Exception as e:
                    print(f"特征拼接错误: {str(e)}")
                    print(f"特征形状: before={before_image.shape}, proj={proj_feature.shape}, after={after_image.shape}")
                    # 错误情况下使用原始嵌入
                    new_input_embeds.append(cur_input_embeds)
                    if labels is not None:
                        new_labels.append(labels[i])
            else:
                # 如果没有image token，直接使用原始嵌入
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[i])
        
        # 填充到最大长度
        max_len = max(embed.shape[0] for embed in new_input_embeds)
        batch_size = len(new_input_embeds)
        
        input_embeds_padded = torch.zeros(
            (batch_size, max_len, new_input_embeds[0].shape[1]),
            device=new_input_embeds[0].device,
            dtype=new_input_embeds[0].dtype
        )
        
        attention_mask_padded = torch.zeros(
            (batch_size, max_len),
            device=input_ids.device, dtype=torch.bool
        )
        
        if labels is not None:
            labels_padded = torch.full(
                (batch_size, max_len),
                self.config.get('ignore_index', -100),
                device=labels.device, dtype=labels.dtype
            )
        
        for i in range(batch_size):
            cur_len = new_input_embeds[i].shape[0]
            input_embeds_padded[i, :cur_len] = new_input_embeds[i]
            attention_mask_padded[i, :cur_len] = 1
            
            if labels is not None:
                labels_padded[i, :cur_len] = new_labels[i]
        
        # 直接调用语言模型部分的forward方法，而不是整个LLaVA模型
        model_outputs = self.llava_model.get_model().forward(
            input_ids=None,  # 我们使用input_embeds
            attention_mask=attention_mask_padded,
            inputs_embeds=input_embeds_padded,
            # 不传入labels，而是后面手动计算损失
        )
        
        # 获取最后一层的隐藏状态
        last_hidden_state = model_outputs.last_hidden_state
        
        # 通过LLaVA模型的lm_head投影层将隐藏状态转换为logits
        logits = self.llava_model.lm_head(last_hidden_state)
        
        # 创建一个包含logits和手动计算loss的自定义输出对象
        class CustomOutput:
            def __init__(self, logits, loss=None, fusion_result=None):
                self.logits = logits
                self.loss = loss
                self.fusion_result = fusion_result
        
        # 手动计算交叉熵损失
        loss = None
        if labels is not None:
            # 使用交叉熵损失函数
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.config.get('ignore_index', -100))
            
            # 计算损失，需要将logits形状从 [batch_size, seq_len, vocab_size] 转换为 [batch_size*seq_len, vocab_size]
            # 标签形状从 [batch_size, seq_len] 转换为 [batch_size*seq_len]
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels_padded.view(-1)
            
            # 计算损失
            loss = loss_fct(logits_flat, labels_flat)
        
        # 创建自定义输出对象
        outputs = CustomOutput(logits, loss, fusion_result)
        
        return outputs

    def generate(self, images, prompts, **generate_kwargs):
        """生成回答"""
        if not self.initialized:
            self.initialize()
        
        # 处理输入
        input_ids = []
        for prompt in prompts:
            # 在提示中添加image token
            if "<image>" not in prompt:
                prompt = "<image>" + prompt
            
            # 编码提示
            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config['training_config']['max_length'])
            input_ids.append(encoded.input_ids)
        
        # 堆叠输入
        input_ids = torch.cat(input_ids, dim=0).to(self.llava_model.device)
        
        # 生成回答
        with torch.no_grad():
            try:
                outputs = self.llava_model.generate(
                    input_ids=input_ids,
                    images=images,
                    **generate_kwargs
                )
            except AttributeError as e:
                # 打印详细错误信息以定位问题
                print(f"生成回答时出错: {str(e)}")
                print(f"images是否为None: {images is None}")
                if images is not None:
                    print(f"images形状: {images.shape}")
                print(f"input_ids是否为None: {input_ids is None}")
                if input_ids is not None:
                    print(f"input_ids形状: {input_ids.shape}")
                    print(input_ids)
                print(f"输入提示: {prompts}")
                # 为了避免程序中断，返回空列表或默认回答
                return ["生成回答时出现错误"]
        
        # 解码回答
        answers = []
        for output in outputs:
            answer = self.tokenizer.decode(output, skip_special_tokens=True)
            # 移除提示部分
            if answers and "<image>" in prompts[i]:
                answer = answer.replace(prompts[i].replace("<image>", ""), "").strip()
            answers.append(answer)
        
        return answers

if __name__ == "__main__":
    # 测试LLaVAOCRModel
    with open("config.json", "r") as f:
        config = json.load(f)
    
    model = LLaVAOCRModel(config)
    model.initialize()
    
    print("LLaVA-OCR模型初始化成功")