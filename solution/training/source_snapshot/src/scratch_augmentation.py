import torch


def single_modality_mixup(batch, modality, coefficient, permutation):
    mask_names = [name for name in batch if name.endswith("_mask")]
    if mask_names != [modality + "_mask"] or not torch.all(batch[modality + "_mask"] == 1):
        raise ValueError("Single-modality mixup requires exactly one fully available modality")
    if not 0 <= coefficient <= 1:
        raise ValueError("Mixing coefficient must be in [0, 1]")
    result = dict(batch)
    result[modality] = coefficient * batch[modality] + (1 - coefficient) * batch[modality][permutation]
    return result
