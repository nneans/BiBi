"""Building blocks of the post-training (train.py, prepare_data.py): data isolation and native ELECTRA RTD.

review_families / canonical_text_rows / diverse_sample keep a review family, and one text, in one split only;
select_mask / corrupt_tokens / replaced_token_loss / NativeGenerator follow google-research/electra (15% of
content tokens, a small generator sharing the token table, loss on the discriminator's token decisions).
"""
from collections import defaultdict, deque
import hashlib
import copy
import random

import torch
from torch import nn
from torch.nn import functional as F
from transformers import ElectraModel
from transformers.models.electra.modeling_electra import ElectraGeneratorPredictions


def review_families(rows):
    """Original review + follow-up links only.

    Identical text is deliberately not a link: chaining through common phrases
    ("맛있게 잘 먹었습니다") merged ~98k unrelated reviews into one family in v1.
    Duplicate text is handled separately by canonical_text_rows.
    """
    parent = {}

    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in rows:
        key = 'r:' + row['review_id']
        root(key)
        for field, prefix in [('review_group_id', 'g:'), ('related_review_id', 'r:')]:
            if row.get(field):
                union(key, prefix + row[field])
    return {r['review_id']: root('r:' + r['review_id']) for r in rows}


def canonical_text_rows(rows, seed):
    """Keep one deterministic copy per text_hash, so a text can never reach two splits."""
    best = {}
    for index, row in enumerate(rows):
        rank = hashlib.sha256(f'{seed}:{row["review_id"]}'.encode()).hexdigest()
        current = best.get(row['text_hash'])
        if current is None or rank < current[0]:
            best[row['text_hash']] = (rank, index)
    return {index for _, index in best.values()}


def diverse_sample(rows, count, seed):
    """Round-robin products, preferring scarce products for shared duplicate families."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['product_id']].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    products = sorted(groups, key=lambda p: (len(groups[p]), p))
    queues = {p: deque(groups[p]) for p in products}
    chosen, used = [], set()
    while len(chosen) < count:
        progress = False
        for product in products:
            queue = queues[product]
            while queue:
                row = queue.popleft()
                if row['family'] in used:
                    continue
                used.add(row['family'])
                chosen.append(row)
                progress = True
                break
            if len(chosen) == count:
                return chosen
        if not progress:
            raise ValueError(f'Only {len(chosen)} independent review families available for {count} samples')
    return chosen


def select_mask(eligible, probability, generator):
    if not bool(eligible.any(-1).all()):
        raise ValueError('Every review needs a content token')
    chosen = (torch.rand(eligible.shape, device=eligible.device, generator=generator) < probability) & eligible
    missing = ~chosen.any(-1)
    fallback = torch.multinomial(eligible.float(), 1, generator=generator).squeeze(-1)
    rows = torch.arange(eligible.shape[0], device=eligible.device)
    chosen[rows[missing], fallback[missing]] = True
    return chosen


def corrupt_tokens(original, chosen, replacements):
    if replacements.ndim != 1 or replacements.numel() != int(chosen.sum()):
        raise ValueError('Replacement count must match selected positions')
    corrupted = original.clone()
    corrupted[chosen] = replacements
    return corrupted, (corrupted != original).float()


def replaced_token_loss(logits, truth, eligible):
    if not bool(eligible.any()):
        raise ValueError('No content positions for token detection loss')
    return F.binary_cross_entropy_with_logits(logits.float()[eligible], truth.float()[eligible])


class NativeGenerator(nn.Module):
    """Small native-vocabulary MLM; fresh body, shared pretrained token embeddings."""
    def __init__(self, discriminator, hidden_size=256, layers=12, heads=4):
        super().__init__()
        config = copy.deepcopy(discriminator.config)
        config.hidden_size = hidden_size
        config.intermediate_size = hidden_size * 4
        config.num_hidden_layers = layers
        config.num_attention_heads = heads
        self.model = ElectraModel(config)
        source = discriminator.electra.embeddings
        self.model.embeddings.word_embeddings = source.word_embeddings
        for name in ['position_embeddings', 'token_type_embeddings', 'LayerNorm']:
            getattr(self.model.embeddings, name).load_state_dict(getattr(source, name).state_dict())
        self.prediction = ElectraGeneratorPredictions(config)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, input_ids, attention_mask, chosen):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return F.linear(self.prediction(hidden[chosen]), self.model.embeddings.word_embeddings.weight, self.bias)
