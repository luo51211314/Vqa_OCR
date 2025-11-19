import torch
import torch.nn as nn
import json
import sys
import os
# 设置环境变量，处理CLIP权重缓存
# 检查本地缓存路径是否有CLIP模型
local_clip_cache_path = "/root/.cache/huggingface/hub"
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = local_clip_cache_path

# 先尝试使用离线模式
os.environ["HF_HUB_OFFLINE"] = "1"

# 如果本地没有缓存，再设置使用hf-mirror镜像站
try:
    # 检查CLIP相关缓存是否存在
    if not os.path.exists(local_clip_cache_path) or len(os.listdir(local_clip_cache_path)) == 0:
        print("本地CLIP缓存不存在或为空，设置使用hf-mirror镜像站")
        # 取消离线模式限制
        os.environ.pop("HF_HUB_OFFLINE", None)
        # 设置hf-mirror镜像站
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_CACHE"] = local_clip_cache_path
except Exception as e:
    print(f"检查缓存时出错: {e}，继续使用离线模式")
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
    def __init__(self, config, llava_model=None, tokenizer=None, ocr_model=None):
        super().__init__()
        self.config = config
        self.llava_model = llava_model
        self.tokenizer = tokenizer
        self.ocr_model = ocr_model
        self.image_processor = None
        self.fused_projector = None
        self.initialized = False
        
        # 在实例化时就完成初始化（如果提供了必要的模型）
        self.initialize()
        
    def initialize(self):
        """初始化模型、分词器和融合投影器，使用从外部传入的模型"""
        if not self.initialized:
            # 检查必要的模型是否已传入
            if self.llava_model is None or self.tokenizer is None:
                raise ValueError("必须从外部传入llava_model和tokenizer")
            
            # 保存model_config以便在图像处理中使用
            self.model_config = self.llava_model.config
            
            # 尝试获取image_processor
            try:
                # 从llava_model中获取image_processor或使用默认配置
                if hasattr(self.llava_model, 'get_vision_tower') and self.llava_model.get_vision_tower() is not None:
                    self.image_processor = self.llava_model.get_vision_tower().image_processor
                else:
                    # 如果无法获取，使用配置中的信息创建一个
                    from transformers import CLIPImageProcessor
                    self.image_processor = CLIPImageProcessor.from_pretrained(
                        self.config['model_config'].get('vision_tower_path', 'openai/clip-vit-large-patch14')
                    )
            except Exception as e:
                print(f"获取image_processor时出错: {e}")
                self.image_processor = None
            
            # 初始化融合投影器，直接传入llava_model、tokenizer和ocr_model
            self.fused_projector = FusedVisionProjector(self.config, llava_model=self.llava_model, tokenizer=self.tokenizer, ocr_model=self.ocr_model)
            
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
        """配置训练参数，冻结不需要训练的模块，明确区分三类参数"""
        # 初始化参数分类
        self.frozen_params = []  # 冻结参数列表
        self.new_params = []     # 新增模块参数列表 (融合器和对齐模块)
        self.unfrozen_params = []  # 解冻参数列表
        
        # 1. 冻结参数 - 视觉编码器
        vision_tower = self.llava_model.get_vision_tower()
        if vision_tower is not None:
            for name, param in vision_tower.named_parameters():
                param.requires_grad = False
                self.frozen_params.append(name)
        
        # 1. 冻结参数 - LLM主体参数（除了LoRA和mm_projector部分）
        if not self.config['lora_config']['lora_enable']:
            # 如果没有使用LoRA，冻结LLM的所有参数
            for name, param in self.llava_model.named_parameters():
                if 'mm_projector' not in name and name not in self.frozen_params:
                    param.requires_grad = False
                    self.frozen_params.append(name)
        
        # 2. 新增模块参数 - fused_projector中的融合器和对齐模块
        # 特别关注ocr_text_projector, fusion_projector和fusion_lm_projector
        for name, param in self.fused_projector.named_parameters():
            param.requires_grad = True
            # 添加完整的参数路径
            full_name = f'fused_projector.{name}'
            self.new_params.append(full_name)
        

    
    def save_new_params(self, save_path):
        """保存融合器和对齐模块的参数（new_params）到指定路径
        
        Args:
            save_path: 保存路径
        """
        # 收集需要保存的参数
        state_dict = {}
        
        # 收集fused_projector中的所有参数（ocr_text_projector, fusion_projector, fusion_lm_projector）
        for name, param in self.fused_projector.named_parameters():
            state_dict[f'fused_projector.{name}'] = param.data
        
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存参数
        torch.save(state_dict, save_path)
        print(f"已保存融合器和对齐模块参数到: {save_path}")
        print(f"保存的参数数量: {len(state_dict)}")
        
    def load_new_params(self, load_path):
        """从指定路径加载融合器和对齐模块的参数（new_params）
        
        Args:
            load_path: 加载路径
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"参数文件不存在: {load_path}")
        
        # 加载参数
        state_dict = torch.load(load_path, map_location='cpu')
        print(f"已加载融合器和对齐模块参数，数量: {len(state_dict)}")
        
        # 更新参数
        for name, param in state_dict.items():
            if name.startswith('fused_projector.'):
                # 移除前缀获取实际参数名
                param_name = name[len('fused_projector.'):]
                # 更新fused_projector中的参数
                if hasattr(self.fused_projector, param_name):
                    getattr(self.fused_projector, param_name).data = param
                else:
                    # 处理嵌套模块的参数
                    parts = param_name.split('.')
                    module = self.fused_projector
                    for part in parts[:-1]:
                        module = getattr(module, part)
                    setattr(module, parts[-1], nn.Parameter(param))
        
        print("融合器和对齐模块参数加载完成")
        
    def set_pretrain_mode(self):
        """设置预训练模式：冻结语言模型，只训练融合器和对齐模块"""
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 只解冻融合器和对齐模块
        for name, param in self.fused_projector.named_parameters():
            if ('ocr_text_projector' in name or 'fusion_projector' in name or 
                'fusion_lm_projector' in name):
                param.requires_grad = True
        
        # 搜索并打印所有可训练参数
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        
        print(f"===== 预训练阶段可训练参数 ====")
        print(f"可训练参数总数: {len(trainable_params)}")
        for param_name in trainable_params[:5]:  # 只打印前5个示例
            print(f"  - {param_name}")
        
        print("已设置为预训练模式：冻结语言模型，只训练融合器和对齐模块")
    
    def set_finetune_mode(self):
        """设置微调模式：冻结视觉编码器和mm_projector层，只解冻new_params部分"""
        # 1. 冻结参数 - 视觉编码器
        vision_tower = self.llava_model.get_vision_tower()
        if vision_tower is not None:
            for name, param in vision_tower.named_parameters():
                param.requires_grad = False
        
        # 2. 冻结参数 - mm_projector层
        for name, param in self.llava_model.named_parameters():
            if 'mm_projector' in name:
                param.requires_grad = False
        
        # 搜索并打印所有可训练参数，对llm lora参数只取一个代表
        trainable_params = []
        lora_params_seen = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                # 处理llm lora参数，只保留一个代表
                if 'lora_' in name:
                    # 提取lora参数的基础名称作为代表
                    lora_base_name = '_'.join(name.split('_')[:3])  # 获取类似 'lora_A_layer' 的前缀
                    if lora_base_name not in lora_params_seen:
                        lora_params_seen.add(lora_base_name)
                        trainable_params.append(f"{lora_base_name}_... (代表所有同类lora参数)")
                else:
                    trainable_params.append(name)
        
        print(f"===== 微调阶段可训练参数 ====")
        print(f"可训练参数总数: {len(trainable_params)}")
        for param_name in trainable_params[:5]:  # 只打印前5个示例
            print(f"  - {param_name}")
    
    def forward(self, images, input_ids, attention_mask=None, labels=None):
        """前向传播"""
        
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
        
        # 直接调用语言模型部分的forward方法，使用完整的参数列表
        outputs = self.llava_model.forward(
            input_ids=None,  # 我们使用input_embeds
            attention_mask=attention_mask_padded,
            inputs_embeds=input_embeds_padded,
            labels=labels_padded,
            return_dict=True
        )
        
        return outputs

    def generate(self, image, prompt, **generate_kwargs):
        """生成回答，与forward方法保持一致，使用fused_projector处理图像"""
        
        # 处理输入 - 现在接受单个prompt
        # 在提示中添加image token
        if "<image>" not in prompt:
            prompt = "<image>" + prompt
        
        # 编码提示
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config['training_config']['max_length'])
        
        # 调试信息
        print(f"Debug: Encoded input_ids shape: {encoded.input_ids.shape}, dtype: {encoded.input_ids.dtype}")
        print(f"Debug: Prompt content: {prompt}")
        
        # 检查input_ids的具体内容和范围
        input_ids_np = encoded.input_ids.cpu().numpy()
        print(f"Debug: input_ids content: {input_ids_np}")
        print(f"Debug: input_ids min: {input_ids_np.min()}, max: {input_ids_np.max()}")
        
        # 检查是否有超出词汇表范围的token
        vocab_size = self.tokenizer.vocab_size
        print(f"Debug: Tokenizer vocab size: {vocab_size}")
        if input_ids_np.max() >= vocab_size:
            print(f"Warning: Found token id {input_ids_np.max()} which exceeds vocab size {vocab_size}")
        
        # 使用更可靠的方式获取设备 - 尝试多种方法
        try:
            # 优先使用模型的第一个参数的设备
            device = next(self.llava_model.parameters()).device
        except Exception as e:
            print(f"Error getting model device: {str(e)}")
            # 备用方案：使用cuda:0
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"Debug: Using device: {device}")
        
        # 移动input_ids到设备时添加错误处理
        try:
            # 先检查CUDA是否可用且工作正常
            if device.type == 'cuda':
                print(f"Debug: CUDA available: {torch.cuda.is_available()}")
                print(f"Debug: CUDA device count: {torch.cuda.device_count()}")
                print(f"Debug: Current CUDA device: {torch.cuda.current_device()}")
                
                # 尝试一个简单的CUDA操作来验证设备是否正常
                try:
                    test_tensor = torch.tensor([1.0], device='cuda')
                    print(f"Debug: CUDA test successful")
                except Exception as cuda_test_err:
                    print(f"Debug: CUDA test failed: {cuda_test_err}")
            
            # 尝试移动input_ids到设备
            print(f"Debug: Attempting to move input_ids to {device}")
            input_ids = encoded.input_ids.to(device)
            print(f"Debug: Successfully moved input_ids to {device}")
            
            # 验证移动后的input_ids
            print(f"Debug: Moved input_ids device: {input_ids.device}")
        except Exception as e:
            print(f"Error moving input_ids to device: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 尝试使用CPU作为备选
            print("Falling back to CPU processing")
            device = torch.device("cpu")
            input_ids = encoded.input_ids.to(device)
            print(f"Debug: Fallback to CPU successful")
        
        # 使用fused_projector处理图像，与forward方法保持一致
        with torch.no_grad():
            print(f"Debug: Processing image, device={device}")
            
            # 详细检查图像类型和属性
            print(f"Debug: Image type: {type(image)}")
            if hasattr(image, 'shape'):
                print(f"Debug: Image shape: {image.shape}")
            if hasattr(image, 'dtype'):
                print(f"Debug: Image dtype: {image.dtype}")
            if torch.is_tensor(image):
                print(f"Debug: Image device before moving: {image.device}")
            
            # 确保图像张量在正确的设备上
            try:
                # 检查image是否为tensor并移动到设备
                if isinstance(image, torch.Tensor):
                    image = image.to(device)
                    print(f"Debug: Image tensor moved to {device}")
            except Exception as e:
                print(f"Error moving image to device: {str(e)}")
            
            # 获取融合特征 - 现在接受单个image
            # 将单个image包装成列表以便fused_projector处理
            try:
                fusion_result = self.fused_projector([image])
                projected_features = fusion_result['projected_features']
                print(f"Debug: Fused projector processed successfully")
            except Exception as e:
                print(f"Error in fused_projector: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # 单个图像的batch_size为1
            batch_size = 1
            
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
            attention_mask = []
            
            for i in range(batch_size):
                # 获取当前样本的输入嵌入
                cur_input_embeds = self.llava_model.get_model().embed_tokens(input_ids[i])
                
                # 创建注意力掩码 - 直接使用input_ids的设备，避免额外的设备指定
                cur_attention_mask = torch.ones_like(input_ids[i], dtype=torch.bool)
                
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
                        
                        # 更新注意力掩码
                        before_mask = cur_attention_mask[:pos]
                        after_mask = cur_attention_mask[pos+1:]
                        proj_mask = torch.ones(proj_feature.shape[0], dtype=torch.bool, device=cur_attention_mask.device)
                        fused_mask = torch.cat([before_mask, proj_mask, after_mask], dim=0)
                        
                        new_input_embeds.append(fused_embeds)
                        attention_mask.append(fused_mask)
                    except Exception as e:
                        print(f"特征拼接错误: {str(e)}")
                        # 错误情况下使用原始嵌入
                        new_input_embeds.append(cur_input_embeds)
                        attention_mask.append(cur_attention_mask)
                else:
                    # 如果没有image token，直接使用原始嵌入
                    new_input_embeds.append(cur_input_embeds)
                    attention_mask.append(cur_attention_mask)
            
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
            
            for i in range(batch_size):
                cur_len = new_input_embeds[i].shape[0]
                input_embeds_padded[i, :cur_len] = new_input_embeds[i]
                attention_mask_padded[i, :cur_len] = attention_mask[i]
            
            # 使用处理后的嵌入进行生成，不直接传递images
            # 设置pad_token_id，如果不存在则使用eos_token_id
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            try:
                # 使用forward方法并实现自定义采样逻辑
                # 从generate_kwargs中提取采样参数
                temperature = generate_kwargs.get('temperature', 0.7)
                top_p = generate_kwargs.get('top_p', 0.9)
                max_new_tokens = generate_kwargs.get('max_new_tokens', 100)
                
                # 初始化输出序列
                batch_size = input_embeds_padded.shape[0]
                device = input_embeds_padded.device
                
                # 初始化past_key_values
                past_key_values = None
                
                # 初始化当前输入嵌入
                current_embeds = input_embeds_padded
                current_attention_mask = attention_mask_padded
                
                # 初始化生成的token序列
                generated_tokens = []
                
                # 检查input_embeds中是否包含NaN或inf值
                if torch.isnan(input_embeds_padded).any():
                    print(f"Debug: input_embeds contains NaN values")
                elif torch.isinf(input_embeds_padded).any():
                    print(f"Debug: input_embeds contains inf values")
                else:
                    print(f"Debug: input_embeds does not contain NaN or inf values")
                
                # 自回归生成
                for step in range(max_new_tokens):
                    # 调用forward方法获取logits
                    with torch.no_grad():
                        outputs = self.llava_model.forward(
                            attention_mask=current_attention_mask,
                            inputs_embeds=current_embeds,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True
                        )
                        # 检查logits中是否包含nan值
                        if torch.isnan(outputs.logits).any():
                            print(f"\n=== 检测到NaN值的问答对 ===")
                            print(f"Prompt: {prompt}")
                            # 打印完整token信息
                            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                            print(f"Input tokens: {input_tokens}")
                            print(f"token 位置: {step}")
                            # 打印logits信息（限制显示长度）
                            print(f"Logits shape: {outputs.logits.shape}")
                            print(f"Logits sample (first few elements): {outputs.logits[0, 0, :5] if outputs.logits.ndim >= 3 else outputs.logits[:5]}")
                            print(f"是否包含NaN: {torch.isnan(outputs.logits).any().item()}")
                            print(f"是否包含inf: {torch.isinf(outputs.logits).any().item()}")
                            print(f"=============================\n")
                            
                            # 检测到NaN，替换为终止符
                            eos_token = torch.tensor([self.tokenizer.eos_token_id], device=device).expand(batch_size)
                            generated_tokens.append(eos_token)
                            break
                    
                    # 获取最后一个token的logits
                    last_logits = outputs.logits[:, -1, :]
                    
                    # 应用温度
                    if temperature > 0:
                        last_logits = last_logits / temperature
                    
                    # 应用top_p采样
                    if top_p < 1.0:
                        # 排序logits
                        sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                        # 计算累积概率
                        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                        # 找到第一个累积概率大于top_p的位置
                        mask = cumulative_probs > top_p
                        # 确保至少保留一个token
                        mask[..., 0] = False
                        # 将被mask的token的logits设置为非常小的值
                        sorted_logits[mask] = -float('inf')
                        # 将处理后的logits放回原位置
                        last_logits = torch.zeros_like(last_logits).scatter_(-1, sorted_indices, sorted_logits)
                    
                    # 计算概率分布并采样
                    probs = torch.softmax(last_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                    
                    # 收集生成的token
                    generated_tokens.append(next_tokens)
                    
                    # 检查是否所有序列都已生成结束符
                    if (next_tokens == self.tokenizer.eos_token_id).all():
                        break
                    
                    # 准备下一轮的输入嵌入
                    # 获取新生成token的嵌入
                    next_token_embeds = self.llava_model.get_model().embed_tokens(next_tokens).unsqueeze(1)
                    
                    # 更新输入嵌入
                    current_embeds = torch.cat([current_embeds, next_token_embeds], dim=1)
                    
                    # 更新注意力掩码
                    new_attention_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
                    current_attention_mask = torch.cat([current_attention_mask, new_attention_mask], dim=1)
                    
                    # 更新past_key_values
                    past_key_values = outputs.past_key_values
                
                # 将生成的token堆叠
                generated_tokens = torch.stack(generated_tokens, dim=1)
                
                # 构建完整的输出序列
                # 注意：这里我们只返回生成的部分，而不是整个序列
                outputs = generated_tokens
            except Exception as e:
                # 打印详细错误信息以定位问题
                print(f"生成回答时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                # 为了避免程序中断，返回默认回答
                return ["生成回答时出现错误"]
        
        # 解码回答
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除提示部分
        if "<image>" in prompt:
            prompt_without_image = prompt.replace("<image>", "")
            if answer.startswith(prompt_without_image):
                answer = answer[len(prompt_without_image):].strip()
        
        return [answer]

if __name__ == "__main__":
    # 测试LLaVAOCRModel
    with open("config.json", "r") as f:
        config = json.load(f)
    
    model = LLaVAOCRModel(config)
    model.initialize()
    
    print("LLaVA-OCR模型初始化成功")