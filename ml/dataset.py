import torch

def dataset_collate_function(batch):
    feature = torch.tensor([data["feature"] for data in batch])
    label = torch.tensor([data["label"] for data in batch])

    return {"feature": feature, "label": label}
