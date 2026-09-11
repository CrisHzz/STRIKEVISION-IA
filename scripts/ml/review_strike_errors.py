"""Export exact annotated frames for a small validation error review."""
import json
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def main():
    run = ROOT / 'outputs/strike-detection/head-v1'
    output = run / 'visual_review'
    output.mkdir(exist_ok=True)
    predictions = [json.loads(line) for line in (run / 'validation_predictions.jsonl').read_text().splitlines()]
    manifest = {r['window_id']: r for r in map(json.loads, (ROOT / 'data/processed/datasets/strike_dataset_v1/binary_windows.jsonl').read_text().splitlines())}
    chosen = []
    for fight in ('aldo_font', 'zhang_joanna_1'):
        for kind, label in [('FP', 'no_strike'), ('FN', 'strike')]:
            candidates = sorted([r for r in predictions if r['fight_id'] == fight and r['label'] == label and r['label'] != r['predicted_label']], key=lambda r:r['p_strike'], reverse=kind=='FP')
            selected = []
            for row in candidates:
                if any(row['video_id'] == s['video_id'] and abs(row['start_frame']-s['start_frame']) < 120 for s in selected):
                    continue
                selected.append(row)
                chosen.append(dict(row, kind=kind, video_path=manifest[row['window_id']]['video_path'], review_id=f'{fight}_{kind}_{len(selected)}'))
                if len(selected) == 2:
                    break
    groups = defaultdict(list)
    for row in chosen:
        groups[row['video_path']].append(row)
    for video, rows in groups.items():
        wanted = {i for row in rows for i in range(row['start_frame'], row['end_frame']+1)}
        frames = {}
        with av.open(str(ROOT / video)) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    picture = frame.to_image()
                    picture.thumbnail((480, 270))
                    frames[index] = picture.copy()
                if index >= max(wanted):
                    break
        for row in rows:
            sequence = [frames[i] for i in range(row['start_frame'], row['end_frame']+1)]
            name = row['review_id']
            sequence[0].save(output / f'{name}.gif', save_all=True, append_images=sequence[1:], duration=round(1000/row['fps']), loop=0)
            sheet = Image.new('RGB', (1440, 620), '#151515')
            draw = ImageDraw.Draw(sheet)
            draw.text((12, 8), f"{name} | label={row['label']} p={row['p_strike']:.4f} | frames {row['start_frame']}-{row['end_frame']}", fill='white')
            for tile, offset in enumerate(np.rint(np.linspace(0,len(sequence)-1,6)).astype(int)):
                px, py = (tile%3)*480, 40+(tile//3)*290
                sheet.paste(sequence[offset], (px, py))
                draw.text((px+8, py+270), f"frame {row['start_frame']+offset}", fill='white')
            sheet.save(output / f'{name}.jpg')
            print(f'Saved {name}', flush=True)
    (output / 'selection.json').write_text(json.dumps(chosen, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
