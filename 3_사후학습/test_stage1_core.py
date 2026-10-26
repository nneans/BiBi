"""Data isolation, corruption truth, and genuinely frozen Stage 2 features."""
import pytest
import torch
from transformers import ElectraConfig, ElectraForPreTraining

from stage1_core import (review_families, canonical_text_rows, diverse_sample, select_mask, corrupt_tokens,
                         replaced_token_loss, NativeGenerator, FrozenEncoder, window_plan)


def test_followup_links_form_one_family_but_identical_text_does_not_chain():
    rows = [dict(review_id='a', text_hash='x', review_group_id='ga', related_review_id=''),
            dict(review_id='b', text_hash='x', review_group_id='gb', related_review_id=''),
            dict(review_id='c', text_hash='y', review_group_id='gb', related_review_id='b'),
            dict(review_id='d', text_hash='y', review_group_id='gd', related_review_id='')]
    families = review_families(rows)
    assert families['b'] == families['c']
    assert len({families['a'], families['b'], families['d']}) == 3


def test_canonical_text_rows_keeps_one_deterministic_copy_per_text():
    rows = [dict(review_id=str(i), text_hash='x' if i < 3 else str(i)) for i in range(6)]
    kept = canonical_text_rows(rows, seed=7)
    assert kept == canonical_text_rows(rows, seed=7)
    assert len(kept) == 4 and len([i for i in kept if rows[i]['text_hash'] == 'x']) == 1


def test_diverse_sample_retains_small_products_without_repeating_families():
    rows = [dict(review_id='a', product_id='small', family='1'),
            dict(review_id='b', product_id='large', family='1'),
            dict(review_id='c', product_id='large', family='2'),
            dict(review_id='d', product_id='large', family='3')]
    chosen = diverse_sample(rows, 3, seed=7)
    assert {r['product_id'] for r in chosen} == {'small', 'large'}
    assert len({r['family'] for r in chosen}) == 3


def test_sample_fails_if_independent_reviews_are_insufficient():
    rows = [dict(review_id='a', product_id='p', family='1')]
    with pytest.raises(ValueError):
        diverse_sample(rows, 2, seed=7)


def test_mask_never_selects_padding_or_special_and_covers_short_reviews():
    eligible = torch.tensor([[False, True, False, False], [False, True, True, False]])
    chosen = select_mask(eligible, .0, torch.Generator().manual_seed(8))
    assert not (chosen & ~eligible).any()
    assert chosen.sum(-1).tolist() == [1, 1]
    assert select_mask(eligible, 1., torch.Generator().manual_seed(8)).tolist() == eligible.tolist()


def test_mask_rejects_empty_content():
    with pytest.raises(ValueError):
        select_mask(torch.zeros(1, 3, dtype=torch.bool), .15, torch.Generator())


def test_sampling_original_token_is_not_labeled_replaced():
    original = torch.tensor([[2, 7, 8, 3, 0]])
    chosen = torch.tensor([[False, True, True, False, False]])
    corrupted, labels = corrupt_tokens(original, chosen, torch.tensor([7, 9]))
    assert corrupted.tolist() == [[2, 7, 9, 3, 0]]
    assert labels.tolist() == [[0., 0., 1., 0., 0.]]
    assert original.tolist() == [[2, 7, 8, 3, 0]]


def test_replacement_count_mismatch_is_rejected():
    with pytest.raises(ValueError):
        corrupt_tokens(torch.tensor([[2, 7, 3]]), torch.tensor([[False, True, False]]), torch.tensor([8, 9]))


def test_rtd_loss_excludes_padding_and_special_token_gradients():
    logits = torch.tensor([[99., 0., 0., -99.]], requires_grad=True)
    truth = torch.tensor([[0., 0., 1., 1.]])
    eligible = torch.tensor([[False, True, True, False]])
    loss = replaced_token_loss(logits, truth, eligible)
    assert float(loss) == pytest.approx(.69314718)
    loss.backward()
    assert logits.grad.tolist() == [[0., .25, -.25, 0.]]


def tiny_discriminator():
    config = ElectraConfig(vocab_size=24, embedding_size=8, hidden_size=8,
                          num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
                          max_position_embeddings=16, pad_token_id=0)
    return ElectraForPreTraining(config)


def test_generator_uses_discriminator_vocabulary_and_shares_only_token_table():
    discriminator = tiny_discriminator()
    generator = NativeGenerator(discriminator, hidden_size=4, layers=1, heads=2)
    assert generator.model.embeddings.word_embeddings.weight is discriminator.electra.embeddings.word_embeddings.weight
    assert generator.model.embeddings.position_embeddings.weight is not discriminator.electra.embeddings.position_embeddings.weight
    ids = torch.tensor([[2, 4, 8, 3]])
    chosen = torch.tensor([[False, True, False, False]])
    logits = generator(ids, torch.ones_like(ids), chosen)
    assert logits.shape == (1, 24)
    torch.nn.functional.cross_entropy(logits, torch.tensor([7])).backward()
    assert discriminator.electra.embeddings.word_embeddings.weight.grad.abs().sum() > 0


def test_frozen_encoder_remains_deterministic_when_outer_heads_enter_train_mode():
    frozen = FrozenEncoder(tiny_discriminator().electra)
    frozen.train()
    assert not frozen.encoder.training
    assert not any(p.requires_grad for p in frozen.parameters())
    ids = torch.tensor([[2, 7, 8, 3]])
    a = frozen(input_ids=ids, attention_mask=torch.ones_like(ids))
    b = frozen(input_ids=ids, attention_mask=torch.ones_like(ids))
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not a.requires_grad


@pytest.mark.parametrize('length', [3, 8, 9, 30, 2502])
def test_window_plan_covers_every_content_token_once_within_limit(length):
    spans, owner = window_plan(length, max_length=8 if length < 100 else 512,
                               overlap=3 if length < 100 else 384)
    capacity = (8 if length < 100 else 512) - 2
    assert all(e - s <= capacity for s, e in spans)
    assert spans[0][0] == 0 and spans[-1][1] == length - 2
    assert len(owner) == length - 2
    assert all(spans[w][0] <= i < spans[w][1] for i, w in enumerate(owner))


def test_window_owner_maximizes_context_on_the_shorter_side():
    spans, owner = window_plan(20, max_length=8, overlap=4)
    for i, w in enumerate(owner):
        room = min(i - spans[w][0], spans[w][1] - 1 - i)
        assert room == max(min(i - s, e - 1 - i) for s, e in spans if s <= i < e)


def test_window_plan_rejects_invalid_settings():
    with pytest.raises(ValueError):
        window_plan(2)
    with pytest.raises(ValueError):
        window_plan(30, max_length=8, overlap=6)


def test_encode_review_within_limit_is_the_plain_forward():
    frozen = FrozenEncoder(tiny_discriminator().electra)
    ids = [2, 7, 8, 9, 3]
    expected = frozen(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, 5, dtype=torch.long))[0]
    torch.testing.assert_close(frozen.encode_review(ids, max_length=16, overlap=4), expected, atol=0, rtol=0)


def test_encode_review_handles_reviews_beyond_position_limit_without_dropping_tokens():
    frozen = FrozenEncoder(tiny_discriminator().electra)
    ids = [2] + [4 + i % 19 for i in range(40)] + [3]
    with pytest.raises(Exception):
        frozen(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long))
    states = frozen.encode_review(ids, max_length=16, overlap=6, window_batch=2)
    assert states.shape == (len(ids), 8)
    spans, owner = window_plan(len(ids), 16, 6)
    s, e = spans[owner[20]]
    window = torch.tensor([[2] + ids[1:-1][s:e] + [3]])
    alone = frozen(input_ids=window, attention_mask=torch.ones_like(window))[0]
    torch.testing.assert_close(states[21], alone[20 - s + 1], atol=0, rtol=0)
    assert not states.requires_grad
