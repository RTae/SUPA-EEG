import torch

from src.encoders.eeg_augmentation import smooth_eeg
from src.models.supaeeg import SUPAEEG, SubjectAwareRouter
from src.utilities import Config, make_model


def test_subject_aware_router_ignores_subject_ids_in_eval_mode():
    router = SubjectAwareRouter(n_subjects=10, n_layers=5)
    subject_ids = torch.tensor([0, 3, 7], dtype=torch.long)

    router.eval()
    with torch.no_grad():
        weights_no_ids = router(None)
        weights_with_ids = router(subject_ids)

    assert weights_no_ids.shape == (1, 5)
    assert weights_with_ids.shape == (3, 5)
    assert torch.allclose(
        weights_with_ids,
        weights_no_ids.expand_as(weights_with_ids),
        atol=1e-6,
    )


def test_supaeeg_encode_image_windows_ignores_subject_ids_in_eval_mode():
    model = SUPAEEG()
    image_layers = torch.randn(4, 5, 3200)
    subject_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        zI_no_ids = model.encode_image_windows(image_layers, subject_ids=None)
        zI_with_ids = model.encode_image_windows(image_layers, subject_ids=subject_ids)

    assert len(zI_no_ids) == 4
    for k in range(4):
        assert zI_no_ids[k].shape == (4, 512)
        assert torch.allclose(zI_no_ids[k], zI_with_ids[k], atol=1e-5)


def test_supaeeg_forward_returns_window_lists_in_train_mode():
    model = SUPAEEG()
    eeg = torch.randn(4, 17, 100)
    image_layers = torch.randn(4, 5, 3200)
    subject_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    model.train()
    zE_list, zI_list = model(eeg, image_layers, subject_ids)

    assert len(zE_list) == 4
    assert len(zI_list) == 4
    for k in range(4):
        assert zE_list[k].shape == (4, 512), f"zE[{k}] shape {zE_list[k].shape}"
        assert zI_list[k].shape == (4, 512), f"zI[{k}] shape {zI_list[k].shape}"


def test_supaeeg_embed_returns_2048():
    model = SUPAEEG()
    eeg = torch.randn(4, 17, 100)

    model.eval()
    with torch.no_grad():
        emb = model.embed(eeg)
        assert emb.shape == (4, 2048), f"embed shape {emb.shape}"

        img_emb = model.encode_image_for_eval(torch.randn(4, 5, 3200))
        assert img_emb.shape == (4, 2048), f"img_emb shape {img_emb.shape}"


def test_make_model_derives_n_layers_from_layer_ids():
    config = Config(n_layers=5, layer_ids=[20, 24, 28])

    model = make_model(config, torch.device("cpu"))

    assert model.router.global_logits.shape == (3,)
    assert model.router.subject_bias.weight.shape == (config.n_subjects, 3)


def test_smooth_eeg_p1_changes_signal():
    eeg = torch.randn(4, 17, 100)
    smoothed = smooth_eeg(eeg, p=1.0)
    assert smoothed.shape == eeg.shape
    assert not torch.allclose(smoothed, eeg)


def test_smooth_eeg_p0_is_identity():
    eeg = torch.randn(4, 17, 100)
    out = smooth_eeg(eeg, p=0.0)
    assert torch.allclose(out, eeg)