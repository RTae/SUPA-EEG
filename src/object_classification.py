import math
import os

import hydra
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset, TensorDataset

from tqdm import tqdm

from dataset import EEGImageNetDataset, build_synthetic_crosspt_loaders
from preprocessing.de_feat_cal import de_feat_cal
from model.eegnet import EEGNet
from model.jepa import (
    EEGJEPA,
    jepa_loss,
    ema_decay_schedule,
    jepa_evaluate,
    load_jepa_checkpoint,
    extract_all_features,
    build_jepa_downstream,
)
from model.eeg_transformer import EEGTransformer
from model.llm_encoder import LLMEEGEncoder
from model.mlp import MLP
from model.rgnn import RGNN, get_edge_weight
from model.simple_model import SimpleModel
from trainer import build_label_map, train_classifier, topk_correct
from utilities import get_device, build_optimizer, get_benchmark_split

SIMPLE_MODELS = {"svm", "rf", "knn", "dt", "ridge"}


def model_init(cfg: DictConfig, num_classes: int, device: torch.device) -> object:
    name = cfg.model.name.lower()
    if cfg.model.type == "simple":
        model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
        return SimpleModel(name, **model_params)
    if name == "eegnet":
        return EEGNet(cfg, num_classes)
    if name == "mlp":
        return MLP(cfg, num_classes)
    if name == "rgnn":
        edge_index, edge_weight = get_edge_weight()
        return RGNN(device, 62, edge_weight, edge_index, 5, 200, num_classes, 2)
    if name == "eeg_transformer":
        return EEGTransformer(cfg)
    if name == "llm_encoder":
        return LLMEEGEncoder(cfg)
    if name == "jepa":
        return EEGJEPA(
            n_channels=int(cfg.model.get("n_channels", 62)),
            seq_len=int(cfg.model.get("seq_len", 400)),
            patch_len=cfg.model.patch_len,
            embed_dim=cfg.model.embed_dim,
            enc_depth=cfg.model.enc_depth,
            pred_depth=cfg.model.pred_depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            dropout=cfg.model.dropout,
            mask_ratio=cfg.model.mask_ratio,
            ema_decay=cfg.model.ema_decay,
        )
    raise ValueError(f"Unknown model: {name}")



@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def load_data(cfg, device):
    data = {}
    dataset = EEGImageNetDataset(
        dataset_dir=cfg.dataset_dir,
        subject=-1,
        granularity=cfg.granularity,
        map_location=device,
    )
    eeg_data = np.stack([sample[0].numpy() for sample in dataset], axis=0)
    de_feat = de_feat_cal(eeg_data, -1, cfg.granularity)
    dataset.add_frequency_feat(de_feat)
    all_labels = np.array([sample[1] for sample in dataset])
    train_idx, test_idx = get_benchmark_split(dataset.data, cfg.metric, cfg.subject)
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    combined_labels = np.concatenate([all_labels[train_idx], all_labels[test_idx]])
    label_map = build_label_map(combined_labels)
    num_classes = len(label_map)
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    is_simple = cfg.model.type == "simple"
    data.update(dict(
        use_synthetic=False,
        num_classes=num_classes,
        pretrain_loader=None,
        train_loader=None,
        test_loader=None,
        label_map=label_map,
        train_subset=train_subset,
        test_subset=test_subset,
        all_labels=all_labels,
        de_feat=de_feat,
        train_idx=train_idx,
        test_idx=test_idx,
        is_simple=is_simple,
        dataset=dataset,
    ))
    
    return data

def train_model(cfg, device, model_obj, data, run_dir):
    results = {}
    is_sematic_training = cfg.model.name.lower() in {"jepa", "eeg_transformer", "llm_encoder"}
    if is_sematic_training:
        # Using semantic training (JEPA or EEG-Transformer)
        pass
    else:
        # Other deep models (EEGNet, MLP, RGNN)
        data["dataset"].use_frequency_feat = cfg.model.use_freq
        train_loader = DataLoader(data["train_subset"], batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(data["test_subset"], batch_size=cfg.batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = build_optimizer(model_obj.parameters(), cfg.model.optimizer)
        save_path = os.path.join(run_dir, f"{cfg.model.name}_s{cfg.subject}.pth")
        acc, epoch = train_classifier(
            model_obj, train_loader, test_loader, criterion, optimizer,
            cfg.model.epochs, device, data["label_map"], save_path=save_path,
        )
        results["deep_acc"] = acc
        results["deep_epoch"] = epoch
    return results

def evaluate_model(cfg, device, model_obj, data, run_dir, train_results):
    is_jepa = cfg.model.name.lower() == "jepa"
    is_transformer = cfg.model.name.lower() in {"eeg_transformer", "llm_encoder"}
    is_simple = data.get("is_simple", False)
    if is_simple:
        # Already evaluated in train_model
        return
    elif is_jepa or is_transformer:
        downstream_head = train_results["downstream_head"]
        ft_test_loader = train_results["ft_test_loader"]
        eval_encoder = train_results["eval_encoder"]
        eval_lmap = train_results["eval_lmap"]
        test_loss, test_top1, test_top5 = jepa_evaluate(
            eval_encoder, downstream_head, ft_test_loader, device, eval_lmap
        )
        print(f"[eval] test_loss={test_loss:.4f} top1={test_top1:.4f} top5={test_top5:.4f}")
        with open(os.path.join(run_dir, "result.txt"), "a", encoding="utf-8") as f:
            f.write(f"top1={train_results['best_top1']:.4f}\n")
    else:
        # Already evaluated in train_model
        return

def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = get_device()
    data = load_data(cfg, device)
    model_obj = model_init(cfg, data["num_classes"], device)
    if cfg.pretrained_model:
        ckpt_path = os.path.join(cfg.output_dir, cfg.pretrained_model)
        is_jepa = cfg.model.name.lower() == "jepa"
        if is_jepa:
            load_jepa_checkpoint(model_obj, ckpt_path, device)
        else:
            model_obj.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    run_dir = HydraConfig.get().runtime.output_dir
    train_results = train_model(cfg, device, model_obj, data, run_dir)
    evaluate_model(cfg, device, model_obj, data, run_dir, train_results)


if __name__ == "__main__":
    main()
