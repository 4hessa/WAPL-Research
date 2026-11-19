"""Read-only recomputation of the stage-63 sampling and paired one-step outcomes."""
import hashlib
import json
import math
import re
import statistics
from itertools import product
from pathlib import Path

import numpy as np

from tools.audit_pud_reuse import close, digest, read_json, require
from tools.audit_pud_reuse_one_step import check_group, sign_counts, verify_branches


def key(row):
    return row['language'], row['seed']


def validated_chunks(language, split, entry, population):
    path = Path(entry['path'])
    require(path == Path(f'data/ud_{language}_{split}.txt'), 'unexpected source path')
    require(digest(path) == entry['sha256'], f'{language}/{split} source digest')
    blob = path.read_bytes()
    require(len(blob) == entry['utf8_bytes'], 'manifest byte count')
    full, tail = divmod(len(blob), 65)
    require(full == population[split+'_chunks'] and tail == population['discarded_tail_bytes'][split],
            f'{language}/{split} chunk count')
    sentences = blob.decode('utf-8').rstrip('\n').split('\n')
    require(len(set(sentences)) == len(sentences) == entry['kept_sentences'], 'duplicate sentence in split')
    return set(sentences), [blob[65*i:65*(i+1)] for i in range(full)]


def audit_populations(result, plan):
    manifest = read_json('data/multilingual_sources.json')
    require(manifest['release'] == 'r2.15', 'source edition')
    summary = {}
    for lang in plan['languages']:
        population = result['populations'][lang]
        text, chunks = {}, {}
        for split in ('train', 'dev', 'test'):
            text[split], chunks[split] = validated_chunks(lang, split,
                manifest['languages'][lang]['splits'][split], population)
        require(text['train'].isdisjoint(text['dev']) and text['train'].isdisjoint(text['test'])
                and text['dev'].isdisjoint(text['test']), 'duplicate sentence across source splits')
        probe = sorted(np.random.default_rng(6301).choice(len(chunks['dev']), size=64, replace=False).tolist())
        target = sorted(np.random.default_rng(6302).choice(len(chunks['test']), size=256, replace=False).tolist())
        require(probe == population['probe_ids'] and target == population['target_ids'], 'sample indices')
        train_content = set(chunks['train'])
        probe_content = {chunks['dev'][i] for i in probe}
        target_content = [chunks['test'][i] for i in target]
        target_set = set(target_content)
        summary[lang] = dict(
            train_duplicate_occurrences=len(chunks['train']) - len(train_content),
            selected_probe_duplicate_occurrences=len(probe) - len(probe_content),
            selected_target_duplicate_occurrences=len(target) - len(target_set),
            train_probe_shared_unique_chunks=len(train_content & probe_content),
            train_target_shared_unique_chunks=len(train_content & target_set),
            probe_target_shared_unique_chunks=len(probe_content & target_set),
            sampled_probe_indices_with_identical_complete_chunk_in_train=sum(chunks['dev'][i] in train_content for i in probe),
            sampled_target_indices_with_identical_complete_chunk_in_train=sum(c in train_content for c in target_content),
            sampled_target_indices_with_identical_complete_chunk_in_probe=sum(c in probe_content for c in target_content))
    return summary


def rebuild_schedule(seed, population):
    rng = np.random.default_rng(seed)
    warmup = rng.integers(population, size=(32, 4))
    candidates = rng.integers(population, size=(16, 4))
    uniform = rng.integers(4, size=16)
    h = hashlib.sha256(candidates.tobytes() + uniform.tobytes()).hexdigest()
    return warmup, candidates, uniform, h


def check_state_records(result, plan):
    states = list(product(plan['languages'], plan['seeds']))
    coverage = {key(r): r for r in result['coverage']}
    require(set(coverage) == set(states) and len(result['coverage']) == len(states) == 16, 'state grid')
    rows = result['rows']
    require(len(rows) == 256, 'decision count')
    for index, state in enumerate(states):
        lang, seed = state
        cov = coverage[state]
        warmup, indices, _, expected = rebuild_schedule(seed, result['populations'][lang]['train_chunks'])
        require(cov['schedule_sha256'] == expected, 'train schedule digest')
        for f in ('model_sha256', 'optimizer_sha256'):
            require(re.fullmatch('[0-9a-f]{64}', cov['initial_'+f]), 'bad initial state fingerprint')
        trajectory = rows[16*index:16*(index+1)]
        for step, row in enumerate(trajectory):
            require(key(row) == state and row['step'] == step and row['age'] == step%4, 'decision grid')
            require(row['candidate_ids'] == indices[step].tolist(), 'wrong training candidate')
            verify_branches(row)
            for f in ('model_sha256', 'optimizer_sha256'):
                prior = cov['initial_'+f] if step == 0 else trajectory[step-1]['reuse_'+f]
                require(row['parent_'+f] == prior, 'incorrect parent branch')
            if step%4 == 0:
                require(row['agreement'] is True and row['delta'] == 0., 'age-zero control')


def check_summaries(result, plan):
    rows = result['rows']
    summary = result['summary']
    states = list(product(plan['languages'], plan['seeds']))
    per_state = {key(r): r for r in summary['states']}
    require(len(summary['states']) == len(per_state) == 16 and set(states) == set(per_state),
            'state summary grid')
    require(set(summary['languages']) == set(plan['languages']), 'language summary grid')
    groups = [(block, [r for r in rows if key(r) == state]) for state, block in per_state.items()]
    groups += [(block, [r for r in rows if r['language'] == lang])
               for lang, block in summary['languages'].items()]
    for block, obs in groups:
        check_group(block['all'], obs)
        check_group(block['stale'], [r for r in obs if r['age'] != 0])
        require(set(block['by_age']) == {'0', '1', '2', '3'}, 'age buckets')
        for age in range(4):
            check_group(block['by_age'][str(age)], [r for r in obs if r['age'] == age])

    per_language = {}
    for lang, data in summary['languages'].items():
        per_seed = []
        for seed in plan['seeds']:
            stale = [r['reuse_loss'] - r['fresh_loss'] for r in rows if key(r) == (lang, seed)
                     and r['age'] != 0]
            require(len(stale) == 12, 'stale outcomes per seed')
            per_seed.append(statistics.fmean(stale))
        close(statistics.fmean(per_seed), data['stale']['mean_delta'], 'primary equal-seed estimate')
        changed = [r for r in rows if r['language'] == lang and r['reuse_choice'] != r['fresh_choice']]
        equal = len([r for r in rows if r['language'] == lang and r['age'] and r['delta'] == 0.])
        per_language[lang] = dict(stale=96, changed=len(changed), stale_exact_ties=equal,
                                  equal_seed_mean_delta=statistics.fmean(per_seed),
                                  changed_counts_by_sign=sign_counts([r['delta'] for r in changed]))
    return per_language


def main():
    path = Path('results/ud-reuse-transfer.json')
    plan_path = Path('research/ud-reuse-transfer-protocol.json')
    result = read_json(path)
    plan = read_json(plan_path)
    require(digest(plan_path) == '7589c59269e58ecbd75293dd89d204c6bbd74247440ee3d3060e6f27819d5145',
            'published protocol digest')
    require(result['stage'] == plan['stage'] == 63 and result['protocol'] == plan, 'protocol mismatch')
    meta = result['provenance']
    require(meta['torch_threads'] == 1 and meta['openblas_num_threads'] == '1' and
            meta['deterministic'] is True, 'execution configuration')
    necessary = {'research/ud-reuse-transfer-protocol.json', 'data/multilingual_sources.json',
                 'experiments/pud_policy.py', 'experiments/pud_reuse.py', 'experiments/pud_reuse_one_step.py',
                 'experiments/ud_reuse_transfer.py', 'tests/test_ud_reuse_transfer.py'}
    necessary |= {f'data/ud_{lang}_{split}.txt' for lang, split in
                  product(plan['languages'], ['train', 'dev', 'test'])}
    require(necessary <= meta['sha256'].keys(), 'missing code or source provenance')
    for name, sha in meta['sha256'].items():
        require(digest(name) == sha, f'changed input {name}')
    chunk_overlap = audit_populations(result, plan)
    check_state_records(result, plan)
    per_language = check_summaries(result, plan)
    disagreements = sum(v['changed'] for v in per_language.values())
    audit = dict(stage=63, status='passed', protocol_commit='0fbc71e52aa9558d3d15d543e8a0dbc5beb8fafb',
                 states=16, decision_pairs=256, branch_evaluations=512, exact_refresh_controls=64,
                 agreeing_decision_pairs_with_identical_states_and_losses=256-disagreements,
                 all_prepared_split_digests_verified=True, no_exact_sentence_overlap_across_splits=True,
                 sampled_complete_chunk_content_overlap=chunk_overlap,
                 recomputed_sorted_probe_and_target_indices=True, reconstructed_training_candidate_indices=True,
                 schedule_digests_verified=True, cached_parent_state_continuity_verified=True,
                 branch_signed_differences_verified=True, summaries_and_empty_conditionals_recomputed=True,
                 results_by_language=per_language, audit_training_runs=0,
                 result_sha256=digest(path), audit_script_sha256=digest(__file__))
    Path('results/ud-reuse-transfer-audit.json').write_text(json.dumps(audit, indent=2)+'\n')


if __name__ == '__main__':
    main()
