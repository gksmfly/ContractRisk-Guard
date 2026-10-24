# tests/model/test_lora_adapter.py
"""LoRA 어댑터(paper_plan §7-26): 학습 파라미터 수·합친 저장·로드 왕복·지문이 어긋나지 않는지 (2026-10-13).

네트워크·GPU 없이 도는 **무작위 초기화 작은 Electra**로 검증한다.
"""

import torch
from transformers import ElectraConfig, ElectraModel

from backend.model.electra import ArticleMultiLabelElectra

NAMES = ["제6조", "제7조", "제8조"]


def _tiny(tmp_path):
    cfg = ElectraConfig(vocab_size=120, embedding_size=16, hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64)
    ElectraModel(cfg).save_pretrained(tmp_path / "base")
    return str(tmp_path / "base")


def _batch():
    torch.manual_seed(0)
    ids = torch.randint(5, 100, (4, 12))
    return ids, torch.ones_like(ids), torch.zeros_like(ids)


def test_lora_freezes_base_and_trains_few_params(tmp_path):
    torch.manual_seed(0)
    m = ArticleMultiLabelElectra(_tiny(tmp_path), NAMES)
    total_before = sum(p.numel() for p in m.parameters())
    m.enable_lora(r=4, alpha=8)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert 0 < trainable < total_before * 0.5
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert any("lora_" in n for n in names) and any("article_head" in n for n in names)
    assert not any("embeddings" in n and "lora" not in n for n in names)      # 기본 가중치는 얼어 있다


def test_lora_forward_matches_after_merged_save_and_load(tmp_path):
    torch.manual_seed(0)
    m = ArticleMultiLabelElectra(_tiny(tmp_path), NAMES)
    m.enable_lora(r=4, alpha=8)
    # 어댑터가 실제로 출력에 영향을 주도록 B 행렬을 흔든다(초기값 0이라 그대로면 기본 모델과 같다)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.2)
    m.eval()
    ids, mask, tt = _batch()
    with torch.no_grad():
        before = m(ids, mask, tt)
    out = tmp_path / "ckpt"
    out.mkdir()
    m.save(out)
    assert hasattr(m.encoder, "merge_and_unload")                              # 저장이 학습 중인 원본을 합쳐 버리지 않았다
    loaded = ArticleMultiLabelElectra.load(out)                                # 지문 검사 통과 = 합친 가중치와 일치
    loaded.eval()
    with torch.no_grad():
        after = loaded(ids, mask, tt)
    assert torch.allclose(before, after, atol=1e-4), (before - after).abs().max()
    assert not hasattr(loaded.encoder, "merge_and_unload")                     # 서빙은 일반 인코더로 읽는다


def test_plain_model_save_load_unchanged(tmp_path):
    torch.manual_seed(0)
    m = ArticleMultiLabelElectra(_tiny(tmp_path), NAMES).eval()
    ids, mask, tt = _batch()
    with torch.no_grad():
        before = m(ids, mask, tt)
    out = tmp_path / "plain"
    out.mkdir()
    m.save(out)
    loaded = ArticleMultiLabelElectra.load(out).eval()
    with torch.no_grad():
        assert torch.allclose(before, loaded(ids, mask, tt), atol=1e-5)
