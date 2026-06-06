import importlib
import sys
import types


def _stub_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_train_module():
    sys.modules.pop("train", None)

    hydra = _stub_module("hydra")
    hydra.main = lambda *args, **kwargs: (lambda fn: fn)

    torch = _stub_module("torch")
    torch.Tensor = object
    torch.long = "long"
    torch.device = lambda *args, **kwargs: None
    torch.stack = lambda items, dim=0: list(items)
    torch.tensor = lambda items, dtype=None: list(items)
    torch.cuda = types.SimpleNamespace(
        empty_cache=lambda: None,
        is_available=lambda: False,
    )
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False),
    )

    _stub_module("torch.utils")
    data_module = _stub_module("torch.utils.data")

    class Dataset:
        pass

    class ConcatDataset:
        def __init__(self, datasets):
            self.datasets = datasets

    class DataLoader:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    data_module.Dataset = Dataset
    data_module.ConcatDataset = ConcatDataset
    data_module.DataLoader = DataLoader

    tensorboard_module = _stub_module("torch.utils.tensorboard")
    tensorboard_module.SummaryWriter = object

    _stub_module("hydra.core")
    hydra_config = _stub_module("hydra.core.hydra_config")
    hydra_config.HydraConfig = type(
        "HydraConfig",
        (),
        {"get": staticmethod(lambda: types.SimpleNamespace(runtime=types.SimpleNamespace(output_dir="/tmp")))},
    )

    omegaconf = _stub_module("omegaconf")
    omegaconf.DictConfig = object
    omegaconf.OmegaConf = types.SimpleNamespace(
        to_container=lambda value, resolve=True: value,
        to_yaml=lambda value: "",
    )

    _stub_module(
        "loguru",
        logger=types.SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
    )
    _stub_module("src")
    _stub_module("src.dataset", ThingsEEGDataset=object)
    _stub_module("src.encoders")
    _stub_module("src.encoders.vision_encoder", InternViTFeatureLookup=object)
    _stub_module(
        "src.utilities",
        Config=object,
        evaluate=lambda *args, **kwargs: None,
        log_results_table=lambda *args, **kwargs: None,
        make_model=lambda *args, **kwargs: None,
        make_optimizer=lambda *args, **kwargs: None,
        save_checkpoint=lambda *args, **kwargs: None,
        train_one_epoch=lambda *args, **kwargs: None,
    )

    return importlib.import_module("train")


def test_inter_subject_wrapper_preserves_global_subject_ids():
    train = _load_train_module()

    class FakeDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return ("eeg", None, 0, 3, 5, "concept", "image.jpg")

    batch = [
        train._SubjectIDDataset(FakeDataset(), subject_id=2)[0],
        train._SubjectIDDataset(FakeDataset(), subject_id=7)[0],
    ]

    collated = train.collate_fn(batch)

    assert collated["subject_ids"] == [1, 6]
