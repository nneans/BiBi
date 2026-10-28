"""Post-training of KcELECTRA on our reviews: three epochs of native ELECTRA RTD (replaced-token detection).

The original public generator is unavailable. A native-vocabulary small MLM is
bootstrapped for one training-corpus pass with the discriminator entirely frozen,
then both networks train with generator CE + 50 * discriminator token BCE.
Input: train/validation/test.jsonl from prepare_data.py (train.jsonl is not kept in the repo; prepare_data.py rebuilds
it byte for byte). The epoch with the lowest token BCE on the fixed validation corruptions is exported as
frozen-encoder/ (2026-10-23 run: epoch 3, see results.json); every later step loads that folder.
Run from this folder: python train.py (resumable: a crash restarts from the last 1,000-step snapshot).
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['RAYON_NUM_THREADS'] = '1'

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoModelForPreTraining, AutoTokenizer

from rtd_core import NativeGenerator, select_mask, corrupt_tokens, replaced_token_loss
from prepare_data import BASE, SEED, sha

OUT = Path(__file__).resolve().parent
CAPS = [(32, 64), (64, 64), (128, 32), (256, 16), (512, 8)]
# This machine crashes at random under long loads (2026-10-23), so joint training is resumable:
# every CHECKPOINT_EVERY steps the full training state is written to RESUME_STATE, and a restart
# continues from there with the same batch order and RNG streams.
RESUME_STATE = OUT / 'resume' / 'joint.pt'
CHECKPOINT_EVERY = 1000


def save_resume(state):
    RESUME_STATE.parent.mkdir(exist_ok=True)
    tmp = RESUME_STATE.with_suffix('.tmp')
    torch.save(state, tmp)
    os.replace(tmp, RESUME_STATE)


def load_bank(name):
    with (OUT / name).open(encoding='utf-8') as f:
        return {item['review_id']: item for item in map(json.loads, f)}


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_rows(split):
    return [json.loads(line) for line in (OUT / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]


def batches(rows, seed=None):
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        for cap, size in CAPS:
            if len(row['input_ids']) <= cap:
                buckets[cap].append(index)
                break
        else:
            raise ValueError('Full native review exceeds encoder length limit')
    result = []
    rng = random.Random(seed)
    for cap, size in CAPS:
        ids = buckets[cap]
        if seed is not None:
            rng.shuffle(ids)
        result.extend((cap, ids[start:start + size]) for start in range(0, len(ids), size))
    if seed is not None:
        rng.shuffle(result)
    flattened = [i for _, group in result for i in group]
    assert len(flattened) == len(rows) and len(set(flattened)) == len(rows)
    return result


def tensor_batch(rows, indices, cap, tokenizer, bank=None):
    ids = torch.full((len(indices), cap), tokenizer.pad_token_id, dtype=torch.long, device='cuda')
    attention = torch.zeros_like(ids)
    for j, index in enumerate(indices):
        row = rows[index]
        sequence = row['input_ids'] if bank is None else bank[row['review_id']]['corrupted_input_ids']
        ids[j, :len(sequence)] = torch.tensor(sequence, device='cuda')
        attention[j, :len(sequence)] = 1
    # UNK remains an ordinary vocabulary value; structural tokens never change.
    eligible = attention.bool()
    for special in [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id]:
        eligible &= ids != special
    return ids, attention, eligible


def sample_native(logits, tokenizer, rng):
    distribution = logits.detach().float().clone()
    for special in [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id]:
        distribution[:, special] = -torch.inf
    return torch.multinomial(distribution.softmax(-1), 1, generator=rng).squeeze(-1)


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_generator(generator, filename):
    state = {k: v.detach().cpu().contiguous() for k, v in generator.state_dict().items()
             if k != 'model.embeddings.word_embeddings.weight'}
    save_file(state, OUT / filename, metadata={'word_embeddings': 'shared discriminator native token table; not duplicated'})


@torch.inference_mode()
def make_bank(rows, generator, tokenizer, seed, name):
    generator.eval()
    rng = torch.Generator(device='cuda').manual_seed(seed)
    bank = {}
    for cap, indices in batches(rows):
        ids, attention, eligible = tensor_batch(rows, indices, cap, tokenizer)
        chosen = select_mask(eligible, .15, rng)
        masked = ids.clone()
        masked[chosen] = tokenizer.mask_token_id
        logits = generator(masked, attention, chosen)
        fake = sample_native(logits, tokenizer, rng)
        corrupted, truth = corrupt_tokens(ids, chosen, fake)
        for j, index in enumerate(indices):
            row = rows[index]
            size = len(row['input_ids'])
            bank[row['review_id']] = {'review_id': row['review_id'], 'original_input_ids': row['input_ids'],
                'corrupted_input_ids': corrupted[j, :size].cpu().tolist(),
                'chosen_positions': torch.where(chosen[j, :size])[0].cpu().tolist(),
                'replaced_labels': truth[j, :size].int().cpu().tolist()}
    path = OUT / name
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(bank[row['review_id']]) + '\n')
    return bank


@torch.inference_mode()
def evaluate(rows, bank, discriminator, generator, tokenizer, tag):
    discriminator.eval()
    generator.eval()
    labels, scores = [], []
    loss_sum = generator_sum = token_count = masked_count = generator_correct = 0
    for cap, indices in batches(rows):
        corrupted, attention, eligible = tensor_batch(rows, indices, cap, tokenizer, bank)
        truth = torch.zeros_like(corrupted, dtype=torch.float32)
        original = corrupted.clone()
        chosen = torch.zeros_like(eligible)
        for j, index in enumerate(indices):
            row = rows[index]
            item = bank[row['review_id']]
            size = len(row['input_ids'])
            truth[j, :size] = torch.tensor(item['replaced_labels'], device='cuda')
            original[j, :size] = torch.tensor(row['input_ids'], device='cuda')
            chosen[j, item['chosen_positions']] = True
        detection = discriminator(input_ids=corrupted, attention_mask=attention).logits.float()
        loss_sum += float(F.binary_cross_entropy_with_logits(detection[eligible], truth[eligible], reduction='sum'))
        token_count += int(eligible.sum())
        labels.append(truth[eligible].cpu().numpy())
        scores.append(detection[eligible].cpu().numpy())
        masked = original.clone()
        masked[chosen] = tokenizer.mask_token_id
        logits = generator(masked, attention, chosen).float()
        generator_sum += float(F.cross_entropy(logits, original[chosen], reduction='sum'))
        masked_count += int(chosen.sum())
        generator_correct += int((logits.argmax(-1) == original[chosen]).sum())
    labels, scores = np.concatenate(labels).astype(np.int8), np.concatenate(scores)
    pred = scores >= 0
    tp = int(((labels == 1) & pred).sum())
    fp = int(((labels == 0) & pred).sum())
    fn = int(((labels == 1) & ~pred).sum())
    tn = int(((labels == 0) & ~pred).sum())
    metrics = {'reviews': len(rows), 'content_tokens': token_count, 'actually_replaced_tokens': int(labels.sum()),
        'fixed_corruption_token_bce': loss_sum / token_count, 'replaced_precision': tp / max(tp + fp, 1),
        'replaced_recall': tp / max(tp + fn, 1), 'replaced_f1': 2 * tp / max(2 * tp + fp + fn, 1),
        'roc_auc': float(roc_auc_score(labels, scores)), 'average_precision': float(average_precision_score(labels, scores)),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'generator_mlm_ce': generator_sum / masked_count,
        'generator_masked_token_accuracy': generator_correct / masked_count}
    np.savez_compressed(OUT / f'{tag}-predictions.npz', labels=labels, logits=scores)
    return metrics


def save_checkpoint(discriminator, tokenizer, epoch):
    path = OUT / f'epoch-{epoch}'
    discriminator.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    shutil.copy2(BASE / 'vocab.txt', path / 'vocab.txt')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(8)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.perf_counter()
    sampling = json.loads((OUT / 'sampling-report.json').read_text(encoding='utf-8'))
    for split, info in sampling['counts'].items():
        assert sha(OUT / f'{split}.jsonl') == info['file_sha256']
    assert shutil.disk_usage(OUT).free > 3 * 1024 ** 3, 'Need space for checkpoints and frozen export'
    train, validation, test = [read_rows(s) for s in ['train', 'validation', 'test']]
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    discriminator = AutoModelForPreTraining.from_pretrained(str(BASE), local_files_only=True,
        use_safetensors=True, attn_implementation='sdpa').cuda()
    generator = NativeGenerator(discriminator).cuda()
    assert generator.model.embeddings.word_embeddings.weight is discriminator.electra.embeddings.word_embeddings.weight
    initial_digest = model_digest(discriminator)
    config = {'stage': '1_only_domain_adaptive_pretraining', 'base_model': 'beomi/KcELECTRA-base-v2022',
        'base_revision': '568fe985a10c013c8f605e3c80a199413bbe9d00', 'initial_model_file_sha256': sha(BASE / 'model.safetensors'),
        'seed': SEED, 'epochs': 3, 'generator_bootstrap_epochs': 1, 'mask_probability': .15,
        'mask_policy': 'Bernoulli 15%, at least one uniformly chosen eligible token per review',
        'input_unit': 'complete_raw_review', 'truncation': False, 'clause_splitting': False,
        'native_vocab_size': len(tokenizer), 'batches': CAPS, 'precision': 'bf16 training / fp32 evaluation',
        'optimizer': 'fused AdamW', 'encoder_learning_rate': 1e-5, 'generator_learning_rate': 1e-4,
        'generator_bootstrap_learning_rate': 3e-4, 'warmup_fraction': .05, 'weight_decay': .01,
        'gradient_clip_norm': 1., 'objective': 'generator MLM CE + 50 * replaced-token BCE',
        'structural_tokens_excluded_from_mask_loss_and_sampling': ['[PAD]', '[CLS]', '[SEP]', '[MASK]'],
        'generator': {'hidden_size': 256, 'layers': 12, 'attention_heads': 4, 'embedding_size': 768,
                      'initialization': 'fresh small body; shared native pretrained token table; copied native position/type/LN embeddings',
                      'original_generator_checkpoint_available': False},
        'checkpoint_selection': 'lowest fixed-corruption validation BCE among the three trained epochs',
        'semantic_labels_used': False, 'ratings_used': False, 'product_id_supplied_to_encoder': False,
        'data_manifest_sha256': sha(OUT / 'sampling-report.json'),
        'reference': 'https://github.com/google-research/electra',
        'adaptations': ['New generator bootstrap because original KcELECTRA generator is unavailable.',
                        'Whole reviews with dynamic length buckets; no document packing.',
                        'Structural special tokens excluded; unchanged sampled originals remain real-token labels.',
                        'Epoch comparison uses one fixed warmed-generator corruption bank.']}
    save('training-config.json', config)
    bootstrap_done = (OUT / 'generator-bootstrap.safetensors').exists() and (OUT / 'generator-bootstrap-audit.json').exists()
    if bootstrap_done:
        from safetensors.torch import load_file
        missing, unexpected = generator.load_state_dict(load_file(OUT / 'generator-bootstrap.safetensors'), strict=False)
        assert missing == ['model.embeddings.word_embeddings.weight'] and not unexpected, (missing, unexpected)
        print(json.dumps({'event': 'generator_bootstrap_reused'}), flush=True)
    else:
        bootstrap_generator(train, discriminator, generator, tokenizer, initial_digest)
    banks_done = all((OUT / n).exists() for n in ['validation-corruptions.jsonl', 'test-corruptions.jsonl', 'baseline-validation.json'])
    if banks_done:
        validation_bank, test_bank = load_bank('validation-corruptions.jsonl'), load_bank('test-corruptions.jsonl')
        baseline = json.loads((OUT / 'baseline-validation.json').read_text(encoding='utf-8'))
    else:
        validation_bank = make_bank(validation, generator, tokenizer, SEED + 100, 'validation-corruptions.jsonl')
        test_bank = make_bank(test, generator, tokenizer, SEED + 200, 'test-corruptions.jsonl')
        baseline = evaluate(validation, validation_bank, discriminator, generator, tokenizer, 'validation-epoch-0')
        save('baseline-validation.json', baseline)
        print(json.dumps({'event': 'baseline_validation', **baseline}), flush=True)
    bank_sha = {'validation': sha(OUT / 'validation-corruptions.jsonl'), 'test': sha(OUT / 'test-corruptions.jsonl')}
    joint_training(train, validation, test, validation_bank, test_bank, bank_sha, baseline,
                   discriminator, generator, tokenizer, sampling, started)


def bootstrap_generator(train, discriminator, generator, tokenizer, initial_digest):
    print(json.dumps({'event': 'generator_bootstrap_start', 'training_reviews': len(train)}), flush=True)
    discriminator.requires_grad_(False)
    discriminator.eval()
    generator.requires_grad_(True)
    generator.model.embeddings.word_embeddings.weight.requires_grad_(False)
    warm_parameters = [p for p in generator.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(warm_parameters, lr=3e-4, weight_decay=.01, fused=True)
    warm_batches = batches(train, SEED + 1)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min((step + 1) / max(1, int(len(warm_batches) * .05)), max(.1, 1 - step / len(warm_batches))))
    rng = torch.Generator(device='cuda').manual_seed(SEED + 10)
    warm_start = time.perf_counter()
    warm_history = []
    running_sum = running_count = 0
    generator.train()
    for step, (cap, indices) in enumerate(warm_batches, 1):
        ids, attention, eligible = tensor_batch(train, indices, cap, tokenizer)
        chosen = select_mask(eligible, .15, rng)
        masked = ids.clone()
        masked[chosen] = tokenizer.mask_token_id
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = generator(masked, attention, chosen)
            loss = F.cross_entropy(logits.float(), ids[chosen])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite generator bootstrap loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(warm_parameters, 1.)
        optimizer.step()
        schedule.step()
        count = int(chosen.sum())
        running_sum += float(loss) * count
        running_count += count
        if step % 100 == 0 or step == len(warm_batches):
            entry = {'event': 'generator_bootstrap', 'step': step, 'steps': len(warm_batches),
                     'masked_token_ce': running_sum / running_count, 'seconds': time.perf_counter() - warm_start}
            warm_history.append(entry)
            save('generator-bootstrap-history.json', warm_history)
            print(json.dumps(entry), flush=True)
    after_warm_digest = model_digest(discriminator)
    assert initial_digest == after_warm_digest, 'Generator bootstrap altered the pretrained discriminator'
    save_generator(generator, 'generator-bootstrap.safetensors')
    save('generator-bootstrap-audit.json', {'discriminator_digest_before': initial_digest,
        'discriminator_digest_after': after_warm_digest, 'discriminator_unchanged': True,
        'training_reviews': len(train), 'epochs': 1, 'seconds': time.perf_counter() - warm_start,
        'generator_file_sha256': sha(OUT / 'generator-bootstrap.safetensors')})
    del optimizer, schedule


def joint_training(train, validation, test, validation_bank, test_bank, bank_sha, baseline,
                   discriminator, generator, tokenizer, sampling, started):
    discriminator.requires_grad_(True)
    generator.requires_grad_(True)
    disc_parameters = list(discriminator.parameters())
    disc_ids = {id(p) for p in disc_parameters}
    gen_parameters = [p for p in generator.parameters() if id(p) not in disc_ids]
    optimizer = torch.optim.AdamW([{'params': disc_parameters, 'lr': 1e-5},
                                   {'params': gen_parameters, 'lr': 1e-4}], weight_decay=.01, fused=True)
    steps_per_epoch = len(batches(train))
    total_steps = steps_per_epoch * 3
    warmup = max(1, int(total_steps * .05))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min((step + 1) / warmup, max(0, (total_steps - step) / (total_steps - warmup))))
    all_parameters = disc_parameters + gen_parameters
    tracked = {name: p.detach().cpu().clone() for name, p in discriminator.named_parameters()
               if name in ['electra.encoder.layer.0.attention.self.query.weight', 'electra.encoder.layer.11.output.dense.weight']}
    first_gradient = None
    history = []
    best = None
    joint_start = time.perf_counter()
    start_epoch, resume = 1, None
    joint_seconds_before = 0.0
    if RESUME_STATE.exists():
        resume = torch.load(RESUME_STATE, map_location='cpu', weights_only=False)  # load_state_dict moves to the params' device
        discriminator.load_state_dict(resume['discriminator'])
        generator.load_state_dict(resume['generator'])
        optimizer.load_state_dict(resume['optimizer'])
        schedule.load_state_dict(resume['schedule'])
        torch.set_rng_state(resume['cpu_rng'].cpu())
        torch.cuda.set_rng_state(resume['cuda_rng'].cpu())
        history, best, first_gradient, tracked = resume['history'], resume['best'], resume['first_gradient'], resume['tracked']
        start_epoch, joint_seconds_before = resume['epoch'], resume['joint_seconds']
        print(json.dumps({'event': 'resumed', 'epoch': resume['epoch'], 'step': resume['step']}), flush=True)

    def snapshot(epoch, step, acc, rng):
        save_resume({'discriminator': discriminator.state_dict(), 'generator': generator.state_dict(),
                     'optimizer': optimizer.state_dict(), 'schedule': schedule.state_dict(),
                     'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                     'rng': rng.get_state() if rng is not None else None, 'epoch': epoch, 'step': step, 'acc': acc,
                     'history': history, 'best': best, 'first_gradient': first_gradient, 'tracked': tracked,
                     'joint_seconds': joint_seconds_before + time.perf_counter() - joint_start})

    for epoch in range(start_epoch, 4):
        discriminator.train()
        generator.train()
        epoch_start = time.perf_counter()
        order = batches(train, SEED + 1000 + epoch)
        rng = torch.Generator(device='cuda').manual_seed(SEED + 1000 + epoch)
        mlm_sum = rtd_sum = masked_total = content_total = actual_total = review_total = 0
        skip, epoch_seconds_before = 0, 0.0
        if resume is not None and resume['epoch'] == epoch and resume['step'] > 0:
            skip = resume['step']
            rng.set_state(resume['rng'].cpu())
            (mlm_sum, rtd_sum, masked_total, content_total, actual_total, review_total,
             epoch_seconds_before) = resume['acc']
        resume = None
        for step, (cap, indices) in enumerate(order, 1):
            if step <= skip:
                continue
            ids, attention, eligible = tensor_batch(train, indices, cap, tokenizer)
            chosen = select_mask(eligible, .15, rng)
            masked = ids.clone()
            masked[chosen] = tokenizer.mask_token_id
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = generator(masked, attention, chosen)
                mlm = F.cross_entropy(logits.float(), ids[chosen])
                fake = sample_native(logits, tokenizer, rng)
                corrupted, truth = corrupt_tokens(ids, chosen, fake)
                detection = discriminator(input_ids=corrupted, attention_mask=attention).logits
                rtd = replaced_token_loss(detection, truth, eligible)
                loss = mlm + 50 * rtd
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch} step {step}')
            loss.backward()
            if first_gradient is None:
                first_gradient = {'encoder_last_layer': float(discriminator.electra.encoder.layer[-1].output.dense.weight.grad.norm()),
                                  'generator_last_layer': float(generator.model.encoder.layer[-1].output.dense.weight.grad.norm())}
            torch.nn.utils.clip_grad_norm_(all_parameters, 1.)
            optimizer.step()
            schedule.step()
            masked_count, count = int(chosen.sum()), int(eligible.sum())
            mlm_sum += float(mlm) * masked_count
            rtd_sum += float(rtd) * count
            masked_total += masked_count
            content_total += count
            actual_total += int(truth[eligible].sum())
            review_total += len(indices)
            if step % 200 == 0 or step == len(order):
                progress = {'event': 'joint_training', 'epoch': epoch, 'step': step, 'steps': len(order),
                    'reviews_seen': review_total, 'mlm_ce': mlm_sum / masked_total,
                    'rtd_bce': rtd_sum / content_total, 'seconds': epoch_seconds_before + time.perf_counter() - epoch_start}
                save('progress.json', progress)
                print(json.dumps(progress), flush=True)
            if step % CHECKPOINT_EVERY == 0 and step < len(order):
                snapshot(epoch, step, (mlm_sum, rtd_sum, masked_total, content_total, actual_total, review_total,
                                       epoch_seconds_before + time.perf_counter() - epoch_start), rng)
            if os.environ.get('STAGE1_TEST_STOP_AT') and step == int(os.environ['STAGE1_TEST_STOP_AT']):
                print(json.dumps({'event': 'test_stop', 'epoch': epoch, 'step': step}), flush=True)
                sys.exit(3)  # resume self-test only
        assert review_total == len(train)
        validation_metrics = evaluate(validation, validation_bank, discriminator, generator, tokenizer, f'validation-epoch-{epoch}')
        assert sha(OUT / 'validation-corruptions.jsonl') == bank_sha['validation']
        entry = {'epoch': epoch, 'train_reviews': review_total, 'content_tokens': content_total,
            'selected_tokens': masked_total, 'actually_replaced_tokens': actual_total,
            'train_generator_mlm_ce': mlm_sum / masked_total, 'train_discriminator_rtd_bce': rtd_sum / content_total,
            'validation': validation_metrics, 'seconds': epoch_seconds_before + time.perf_counter() - epoch_start,
            'review_id_coverage_sha256': hashlib.sha256('\n'.join(sorted(train[i]['review_id'] for _, group in order for i in group)).encode()).hexdigest()}
        save_checkpoint(discriminator, tokenizer, epoch)
        entry['checkpoint_sha256'] = sha(OUT / f'epoch-{epoch}/model.safetensors')
        history.append(entry)
        save('history.json', history)
        print(json.dumps({'event': 'epoch_complete', **entry}), flush=True)
        if best is None or validation_metrics['fixed_corruption_token_bce'] < best['validation_bce']:
            best = {'epoch': epoch, 'validation_bce': validation_metrics['fixed_corruption_token_bce']}
        save('best.json', best)
        snapshot(epoch + 1, 0, None, None)  # epoch done; a restart begins the next epoch
    save_generator(generator, 'generator-final.safetensors')
    changes = {name: float((dict(discriminator.named_parameters())[name].detach().cpu() - before).norm())
               for name, before in tracked.items()}
    save('training-audit.json', {'first_step_gradient_norms': first_gradient, 'three_epoch_parameter_change_norms': changes,
        'joint_training_seconds': joint_seconds_before + time.perf_counter() - joint_start, 'corruption_bank_sha256': bank_sha,
        'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
        'model_digest_after_epoch3': model_digest(discriminator)})
    # Test set is evaluated only after the validation-based checkpoint decision.
    original = AutoModelForPreTraining.from_pretrained(str(BASE), local_files_only=True,
        use_safetensors=True, attn_implementation='sdpa').cuda()
    baseline_test = evaluate(test, test_bank, original, generator, tokenizer, 'test-baseline')
    del original
    selected = OUT / f'epoch-{best["epoch"]}'
    from safetensors.torch import load_file
    discriminator.load_state_dict(load_file(selected / 'model.safetensors'), strict=True)
    selected_test = evaluate(test, test_bank, discriminator, generator, tokenizer, 'test-selected')
    # Generator metrics above describe the final generator; RTD checkpoint choice
    # and reported before/after RTD comparisons use identical stored corruptions.
    for metric in [baseline_test, selected_test]:
        metric['generator_metrics_reference'] = 'final joint generator; not used in discriminator checkpoint selection'
    export = OUT / 'frozen-encoder'
    discriminator.electra.eval().requires_grad_(False)
    discriminator.electra.save_pretrained(export, safe_serialization=True)
    tokenizer.save_pretrained(export)
    shutil.copy2(BASE / 'vocab.txt', export / 'vocab.txt')
    results = {'stage': '1_food_review_domain_adaptation', 'training_reviews': len(train),
        'training_products': sampling['counts']['train']['products'], 'validation_reviews': len(validation),
        'test_reviews': len(test), 'epochs_completed': 3, 'generator_bootstrap_epochs': 1,
        'selected_epoch': best['epoch'], 'baseline_validation': baseline,
        'selected_validation': history[best['epoch'] - 1]['validation'],
        'baseline_test': baseline_test, 'selected_test': selected_test,
        'elapsed_seconds_including_generator_bootstrap_and_model_saving': time.perf_counter() - started,
        'frozen_encoder_path': str(export), 'frozen_encoder_sha256': sha(export / 'model.safetensors'),
        'frozen_loader': 'load_frozen.py:load_encoder', 'native_vocab_sha256': sha(export / 'vocab.txt'),
        'source_native_vocab_sha256': sha(BASE / 'vocab.txt'), 'corruption_bank_sha256': bank_sha,
        'scope': {'sentiment_training': False, 'relation_training': False, 'clustering_training': False,
                  'external_llm': False, 'semantic_labels_generated': False, 'product_classifier': False},
        'interpretation': ['RTD validation/test measure changed-token detection, not sentiment or aspect/opinion relationship accuracy.',
                           'This adapts the original KcELECTRA discriminator using a new bootstrapped native generator; it does not resume the unavailable original pretraining optimizer/generator state.',
                           'The selected encoder is ready for frozen-feature Stage 2 experiments; task heads require separate training and evaluation.']}
    assert results['native_vocab_sha256'] == results['source_native_vocab_sha256']
    assert sha(OUT / 'test-corruptions.jsonl') == bank_sha['test']
    save('results.json', results)
    print(json.dumps({'event': 'complete', 'selected_epoch': best['epoch'],
        'baseline_test_bce': baseline_test['fixed_corruption_token_bce'],
        'selected_test_bce': selected_test['fixed_corruption_token_bce'],
        'elapsed_seconds': results['elapsed_seconds_including_generator_bootstrap_and_model_saving']}), flush=True)


if __name__ == '__main__':
    main()
