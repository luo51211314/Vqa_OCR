import json, os, tempfile, io, numpy as np, pandas as pd
from PIL import Image
from loaders import VqaDataset

class Dataset(VqaDataset):
    name = "scienceqa"

    def __init__(self, split="validation", **_):
        super().__init__(split)
        # 这里仅示范：读 parquet
        self.data = pd.read_parquet(
            f"/root/autodl-tmp/datasets/scienceqa/{split}-00000-of-00001.parquet"
        )
        self.data = self.data[self.data["image"].notna()].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        choices = []
        if isinstance(row["choices"], np.ndarray) and row["choices"].size > 0:
            choices = [f"{i}. {c}" for i, c in enumerate(row["choices"].tolist())]
        if not choices:
            choices = ["0. True", "1. False"]
        prompt = f"{row['question']}\n选项: {', '.join(choices)}，请只输出选项前的数字"
        answer = int(row["answer"])          # int
        return img, prompt, [str(answer)], {"question_id": row.name}

    @staticmethod
    def metrics(preds, refs):
        from sklearn.metrics import accuracy_score, f1_score
        pred_num = [int(re.search(r"\d", p).group()) if re.search(r"\d", p) else -1 for p in preds]
        ref_num = [int(r[0]) for r in refs]
        return {"accuracy": float(accuracy_score(ref_num, pred_num)),
                "f1": float(f1_score(ref_num, pred_num, average="macro"))}