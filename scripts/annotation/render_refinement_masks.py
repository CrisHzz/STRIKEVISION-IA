"""Render every action window with exactly the joints retained by its masks."""

import argparse
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw

from prepare_round2_refinement import EDGES, read, round_paths, save, sha


def render(round_number):
    _, rgb, destination, _ = round_paths(round_number)
    inventory = read(destination / 'overlay_inventory.json')
    report = read(destination / 'geometry_report.json')
    mask_path = destination / 'geometry_masks.npz'
    mask_hash = sha(mask_path)
    identity_path = destination / 'identity_decisions.json'
    assert report['identity_audit_complete']
    assert report['identity_decisions_sha256'] == sha(identity_path)
    decisions = {row['frame_index']: row['fighters'] for row in read(identity_path)['frames']}
    manifest = read(rgb / 'manifest.json')
    capture = cv2.VideoCapture(manifest['video'])
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    capture.release()
    assert width > 0 and height > 0

    output = destination / 'masked_overlays' / mask_hash
    output.mkdir(parents=True, exist_ok=True)
    names = report['joint_names']
    edges = [(names.index(first), names.index(last)) for first, last in EDGES]
    with np.load(mask_path) as archive:
        frames, coordinates, points = (
            archive[key] for key in ('frame_indices', 'coordinates_px', 'point_mask')
        )
        assert archive['action_ids'].tolist() == [item['action_id'] for item in inventory['actions']]

    actions = []
    for index, action in enumerate(inventory['actions']):
        actor = action['fighter']
        rival = next(name for name in inventory['fighter_counts'] if name != actor)
        start, end = action['review_window']
        sheets = []
        for sheet_index in range(start // 16, end // 16 + 1):
            sheet = manifest['sheets'][sheet_index]
            with Image.open(rgb / sheet['file']) as source:
                original = source.convert('RGB')
            tile_width, tile_height = original.width // 4, original.height // 4
            label_height = 30
            tile_stride = tile_height + label_height
            result = Image.new('RGB', (original.width, 4 * tile_stride))
            first = max(start, sheet['start_frame'])
            last = min(end, sheet['end_frame'])
            for frame in range(first, last + 1):
                offset = frame - start
                assert frames[index, offset] == frame
                tile = frame - sheet['start_frame']
                x, y = tile % 4 * tile_width, tile // 4 * tile_height
                crop = original.crop((x, y, x + tile_width, y + tile_height))
                draw = ImageDraw.Draw(crop)
                result_y = tile // 4 * tile_stride
                labels = []
                for slot, (fighter, color) in enumerate(((actor, '#52ff70'), (rival, '#42b5ff'))):
                    choice = decisions[frame][fighter]
                    valid = points[index, offset, slot]
                    xy = coordinates[index, offset, slot] * (tile_width / width, tile_height / height)
                    for a, b in edges:
                        if valid[a] and valid[b]:
                            draw.line([tuple(xy[a]), tuple(xy[b])], fill=color, width=2)
                    for px, py in xy[valid]:
                        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
                    label = f'{fighter} | {choice["status"]} | {valid.sum()}/17'
                    if choice['status'] == 'verified':
                        label += f' | fila {choice["line_1based"]}'
                    labels.append((label, color))
                result.paste(crop, (x, result_y))
                legend = ImageDraw.Draw(result)
                for slot, (label, color) in enumerate(labels):
                    legend.text((x + 4, result_y + tile_height + slot * 15), label, fill=color)
            filename = f'{index + 1:03d}_{first:06d}_{last:06d}.jpg'
            result.save(output / filename, quality=88)
            sheets.append({'first_frame': first, 'last_frame': last, 'file': filename})
        actions.append({**action, 'identity_review': 'complete', 'masked_sheets': sheets})
    save(output / 'manifest.json', {
        'masks_sha256': mask_hash,
        'identity_decisions_sha256': report['identity_decisions_sha256'],
        'identity_status_counts': dict(Counter(
            choice['status'] for fighters in decisions.values() for choice in fighters.values()
        )),
        'actions': actions,
    })
    print(f'Round {round_number}: {len(actions)} windows, {sum(len(a["masked_sheets"]) for a in actions)} sheets. {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rounds', type=int, choices=range(1, 6), nargs='+')
    for number in parser.parse_args().rounds:
        render(number)
