import importlib, os
from torch.utils.data import DataLoader

def list_datasets():
    return [f[:-3] for f in os.listdir(os.path.dirname(__file__)+"/loaders")
            if f.endswith(".py") and f != "__init__.py"]

def build_dataset(name: str, split: str = "validation", **kw):
    """name = scienceqa / docvqa / gqa / …"""
    module = importlib.import_module(f"loaders.{name}")
    ds = module.Dataset(split=split, **kw)
    return ds

def build_dataloader(name, split="validation", batch_size=1, num_workers=4, **kw):
    ds = build_dataset(name, split, **kw)
    def collate(batch):
        imgs, prompts, answers, extras = zip(*batch)
        return list(imgs), list(prompts), list(answers), list(extras)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, collate_fn=collate)