"""Apply five explicitly visually reviewed labels and migrate immutable features."""
import copy
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ufc_tracker.annotations.dataset import sha256
from ufc_tracker.strike.cache import (context, read_json, atomic_json, digest,
    groups_for, cache_path, video_state, check_shard, load_features)

ROOT = Path(__file__).resolve().parents[2]


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + '.review.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    os.replace(temporary, path)


def main():
    current = context(ROOT)
    old_key = read_json(ROOT / 'outputs/strike-detection/head-v1/run.json')['identity']['cache_key']
    old_identity = read_json(ROOT / 'data/processed/features/strike_x3d_s' / old_key / 'identity.json')
    assert digest(old_identity) == old_key
    assert {k:v for k,v in old_identity.items() if k != 'source_code'} == {
        k:v for k,v in current['identity'].items() if k != 'source_code'}
    old_ctx = {**current, 'identity': old_identity, 'key': old_key}
    # Verify existing source files, complete shards and alignment before any edits.
    for split in ('train', 'validation'):
        load_features(ROOT, old_ctx, split)
    selection = read_json(ROOT / 'outputs/strike-detection/head-v1/visual_review/selection.json')
    decisions = {
        'aldo_font_FP_2': ('strike', 'Visible punch sequence directed at opponent guard/head'),
        'zhang_joanna_1_FP_1': ('strike', 'Visible punches and low kick within annotated interval'),
        'zhang_joanna_1_FP_2': ('strike', 'Visible punch exchange within annotated interval'),
        'zhang_joanna_1_FN_1': ('no_strike', 'All 24 frames reviewed: guard and repositioning; no discernible strike'),
        'zhang_joanna_1_FN_2': ('no_strike', 'All 24 frames reviewed: guard and footwork; no discernible strike'),
    }
    rows = copy.deepcopy(current['rows'])
    by_id = {r['window_id']: r for r in rows}
    changes = []
    for item in selection:
        if item['review_id'] not in decisions:
            continue
        row = by_id[item['window_id']]
        assert row['fight_id'] in current['split']['assignments']['validation']
        assert row['label'] == item['label']
        label, reason = decisions[item['review_id']]
        changes.append(dict(window_id=row['window_id'], source=row['annotation_file'],
            old_label=row['label'], new_label=label, reason=reason,
            evidence=f"outputs/strike-detection/head-v1/visual_review/{item['review_id']}_all.jpg"))
        row['label'] = label
    assert len(changes) == 5
    backup = ROOT / 'outputs/annotation-review-v1'
    backup.mkdir(exist_ok=False)
    dataset = ROOT / 'data/processed/datasets/strike_dataset_v1'
    paths = {ROOT / c['source'] for c in changes} | {
        dataset / 'binary_windows.jsonl', dataset / 'audit.json',
        ROOT / 'configs/data/strike_split_v1.json'}
    for path in paths:
        target = backup / 'before' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    journal = dict(status='prepared', timestamp=datetime.now(timezone.utc).isoformat(),
        authorization='User explicitly authorized visually justified corrections',
        criterion='Visible strike attempt within the window, including blocked strikes; no claim of effective impact',
        changes=changes, old_cache_key=old_key, before_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in paths})
    atomic_json(backup / 'changes.json', journal)
    for relative in {c['source'] for c in changes}:
        path = ROOT / relative
        source_rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        edits = {c['window_id']:c for c in changes if c['source'] == relative}
        for row in source_rows:
            if row['window_id'] in edits:
                assert row['label'] == edits[row['window_id']]['old_label']
                row['label'] = edits[row['window_id']]['new_label']
        write_rows(path, source_rows)
    write_rows(dataset / 'binary_windows.jsonl', rows)
    audit = read_json(dataset / 'audit.json')
    review_rows = [json.loads(line) for line in (dataset / 'review_windows.jsonl').read_text().splitlines()]
    audit['labels'] = dict(Counter(r['label'] for r in rows + review_rows))
    audit['binary_manifest_sha256'] = sha256(dataset / 'binary_windows.jsonl')
    audit['label_review'] = 'outputs/annotation-review-v1/changes.json'
    for change in changes:
        video = by_id[change['window_id']]['video_id']
        audit['sources'][video]['annotation_sha256'] = sha256(ROOT / change['source'])
    atomic_json(dataset / 'audit.json', audit)
    split_info = read_json(ROOT / 'configs/data/strike_split_v1.json')
    split_info['manifest_sha256'] = audit['binary_manifest_sha256']
    split_info['label_review'] = 'outputs/annotation-review-v1/changes.json'
    atomic_json(ROOT / 'configs/data/strike_split_v1.json', split_info)
    new_ctx = context(ROOT)
    target = cache_path(ROOT, new_ctx)
    target.mkdir(exist_ok=False, parents=True)
    atomic_json(target / 'identity.json', new_ctx['identity'])
    for split in ('train', 'validation'):
        old_groups = groups_for(old_ctx, split)
        for video, new_rows in groups_for(new_ctx, split).items():
            original_rows = old_groups[video]
            assert [{k:v for k,v in r.items() if k != 'label'} for r in new_rows] == [
                {k:v for k,v in r.items() if k != 'label'} for r in original_rows]
            source_dir = cache_path(ROOT, old_ctx) / split / video
            state, _ = video_state(source_dir, original_rows, sha256(ROOT / new_rows[0]['video_path']), old_key)
            destination = target / split / video
            destination.mkdir(parents=True)
            for shard in state['shards']:
                old_x = check_shard(source_dir / shard['file'], shard, original_rows)
                with np.load(source_dir / shard['file'], allow_pickle=False) as data:
                    arrays = {key:data[key] for key in data.files}
                arrays['labels'] = np.array([int(r['label']=='strike') for r in new_rows[shard['start']:shard['stop']]], dtype=arrays['labels'].dtype)
                np.savez(destination / shard['file'], **arrays)
                shard['sha256'] = sha256(destination / shard['file'])
                assert np.array_equal(old_x, check_shard(destination / shard['file'], shard, new_rows))
            state['contract']['cache_key'] = new_ctx['key']
            state['label_review'] = 'outputs/annotation-review-v1/changes.json'
            atomic_json(destination / 'index.json', state)
        print(f'Migrated {split}: unchanged embeddings, reviewed labels', flush=True)
    for split in ('train','validation'):
        load_features(ROOT, new_ctx, split)
    journal.update(status='complete', new_cache_key=new_ctx['key'],
        after_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in paths},
        embedding_values_changed=False, test_modified=False, retrained=False)
    atomic_json(backup / 'changes.json', journal)
    print(json.dumps({'corrections':len(changes), 'cache':new_ctx['key'], 'verified':True}))


if __name__ == '__main__':
    main()
