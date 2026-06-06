import torch

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


def test_supaeeg_forward_accepts_subject_ids_in_train_mode():
    model = SUPAEEG()
    eeg = torch.randn(4, 17, 100)
    image_layers = torch.randn(4, 5, 3200)
    subject_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    model.train()
    eeg_emb, image_emb, subj_logits = model(eeg, image_layers, subject_ids)

    assert eeg_emb.shape == (4, 512)
    assert image_emb.shape == (4, 512)
    assert subj_logits is not None and subj_logits.shape == (4, 10)


def test_make_model_derives_n_layers_from_layer_ids():
    config = Config(n_layers=5, layer_ids=[20, 24, 28])

    model = make_model(config, torch.device("cpu"))

    assert model.router.global_logits.shape == (3,)
    assert model.router.subject_bias.weight.shape == (config.n_subjects, 3)