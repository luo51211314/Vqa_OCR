from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset

class VqaDataset(Dataset):
    """
    所有自定义数据集必须继承此类，并实现：
        __len__ / __getitem__ -> Tuple[PIL.Image, str, Any, Any]
        img, prompt, answers, extra
    """
    def __init__(self, split: str = "validation", **kw):
        self.split = split

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        # return  img: PIL.Image
        #         prompt: str  （已含 <image> token）
        #         answers: List[str] 或 int
        #         extra: dict 任意附加信息
        raise NotImplementedError

    @staticmethod
    def metrics(preds: List[str], refs: List[Any]) -> Dict[str, float]:
        """数据集专属评测指标，必须实现"""
        raise NotImplementedError