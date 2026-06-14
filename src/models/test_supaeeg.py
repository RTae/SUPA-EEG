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


def test_supaeeg_encode_image_ignores_subject_ids_in_eval_mode():
    model = SUPAEEG()
    image_layers = torch.randn(4, 5, 3200)
    subject_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        image_emb_no_ids = model.encode_image(image_layers, subject_ids=None)
        image_emb_with_ids = model.encode_image(image_layers, subject_ids=subject_ids)

    assert image_emb_no_ids.shape == (4, 512)
    assert image_emb_with_ids.shape == (4, 512)
    assert torch.allclose(image_emb_no_ids, image_emb_with_ids, atol=1e-5)


def test_supaeeg_forward_returns_single_embeddings():
    model = SUPAEEG()
    eeg = torch.randn(4, 17, 100)
    image_layers = torch.randn(4, 5, 3200)
    subject_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    model.train()
    zE, zI = model(eeg, image_layers, subject_ids)

    assert zE.shape == (4, 512)
    assert zI.shape == (4, 512)


def test_supaeeg_embed_returns_512():
    model = SUPAEEG()
    eeg = torch.randn(4, 17, 100)

    model.eval()
    with torch.no_grad():
        emb = model.embed(eeg)
        assert emb.shape == (4, 512)


def test_make_model_derives_n_layers_from_layer_ids():
    config = Config(n_layers=5, layer_ids=[20, 24, 28])

    model = make_model(config, torch.device("cpu"))

    assert model.router.global_logits.shape == (3,)
    assert model.router.subject_bias.weight.shape == (config.n_subjects, 3)


def test_all_eeg_ablation_encoders_return_expected_shape():
    eeg = torch.randn(2, 17, 100)
    for encoder_type in ("eegproject", "eegnet", "tsconv", "eegconformer", "atm"):
        model = SUPAEEG(eeg_encoder_type=encoder_type)
        assert model.encode_eeg(eeg).shape == (2, 512)


def test_image_layer_modes_select_expected_layers():
    layers = torch.zeros(2, 5, 3200)
    layers[:, 2] = 1.0
    single = SUPAEEG(image_layer_mode="single", image_layer_index=2)
    uniform = SUPAEEG(image_layer_mode="uniform")
    with torch.no_grad():
        assert single.encode_image(layers).shape == (2, 512)
        assert uniform.encode_image(layers).shape == (2, 512)


def test_temporal_compression_accepts_full_rate_input():
    model = SUPAEEG(n_timepoints=1000, temporal_compression=100)
    eeg = torch.randn(2, 17, 1000)
    assert model.encode_eeg(eeg).shape == (2, 512)


def test_smooth_eeg_p1_changes_signal():
    eeg = torch.randn(4, 17, 100)
    smoothed = smooth_eeg(eeg, p=1.0)
    assert smoothed.shape == eeg.shape
    assert not torch.allclose(smoothed, eeg)


def test_smooth_eeg_p0_is_identity():
    eeg = torch.randn(4, 17, 100)
    out = smooth_eeg(eeg, p=0.0)
    assert torch.allclose(out, eeg)
