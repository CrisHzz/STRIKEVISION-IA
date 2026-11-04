from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ReportLab is supplied by the Codex document runtime. Append (rather than
# prepend) it so project libraries keep precedence.
CODEX_SITE_PACKAGES = Path(
    r"C:\Users\Cristian\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages"
)
if CODEX_SITE_PACKAGES.exists():
    sys.path.append(str(CODEX_SITE_PACKAGES))

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
ANNOTATION = ROOT / "data/annotations/strike_actions_zhang_joanna_trial/zhang_joanna_1__zhang_weili_vs_joanna_jedrzejczyk_1__aggressive_women_round4.json"
OUTPUT = ROOT / "output/pdf/zhang_vs_joanna_round4_auditoria_clasificacion.pdf"
ROUND_NUMBER = 4
EXPECTED_FRAGMENTS = 66
REVIEW_DIR = ROOT / "tmp_round4_audit"

LABEL_NAMES = {
    "punch_head": "Puno a la cabeza",
    "punch_body": "Puno al cuerpo",
    "kick_head": "Patada a la cabeza",
    "kick_body": "Patada al cuerpo",
    "kick_leg": "Patada a la pierna",
    "other_strike": "Otro golpe",
}
LABEL_COLORS = {
    "punch_head": colors.HexColor("#8B1E3F"),
    "punch_body": colors.HexColor("#B34726"),
    "kick_head": colors.HexColor("#5C2A9D"),
    "kick_body": colors.HexColor("#235789"),
    "kick_leg": colors.HexColor("#167D62"),
    "other_strike": colors.HexColor("#555B66"),
}
ORDER = ["punch_head", "punch_body", "kick_head", "kick_body", "kick_leg", "other_strike"]


def footer(pdf: canvas.Canvas, page_no: int) -> None:
    width, _ = landscape(letter)
    pdf.setStrokeColor(colors.HexColor("#D8DEE8"))
    pdf.line(28, 20, width - 28, 20)
    pdf.setFillColor(colors.HexColor("#657184"))
    pdf.setFont("Helvetica", 7)
    pdf.drawString(28, 9, f"StrikeVision ML - auditoria visual - Zhang vs Joanna - Round {ROUND_NUMBER}")
    pdf.drawRightString(width - 28, 9, f"Pagina {page_no}")


def fit_text(pdf: canvas.Canvas, text: str, max_width: float, size: float = 8) -> None:
    while size > 5 and stringWidth(text, "Helvetica", size) > max_width:
        size -= 0.25
    pdf.setFont("Helvetica", size)


def draw_frame(pdf: canvas.Canvas, reader: ImageReader, x: float, y: float, w: float, h: float, label: str) -> None:
    pdf.setFillColor(colors.HexColor("#111827"))
    pdf.roundRect(x, y, w, h, 3, fill=1, stroke=0)
    pdf.drawImage(reader, x + 2, y + 14, w - 4, h - 16, preserveAspectRatio=True, anchor="c", mask="auto")
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 6.5)
    pdf.drawCentredString(x + w / 2, y + 4, label)


def main() -> None:
    data = json.loads(ANNOTATION.read_text(encoding="utf-8"))
    actions = data["actions"]
    event_number = {event_id: i + 1 for i, event_id in enumerate(data["completed"])}

    grouped: dict[str, list[dict]] = defaultdict(list)
    for action in actions:
        grouped[action["label"]].append(action)
    for label in grouped:
        grouped[label].sort(key=lambda a: (event_number[a["event_id"]], a["start_frame"], a["attacker"]))

    by_event = defaultdict(list)
    for action in actions:
        by_event[action["event_id"]].append(action)
    simultaneous = set()
    for event_id, event_actions in by_event.items():
        for i, left in enumerate(event_actions):
            for right in event_actions[i + 1 :]:
                overlap = max(left["start_frame"], right["start_frame"]) <= min(left["end_frame"], right["end_frame"])
                if left["attacker"] != right["attacker"] and overlap:
                    simultaneous.add(left["action_id"])
                    simultaneous.add(right["action_id"])

    event_sheets = {
        number: ImageReader(str(REVIEW_DIR / f"event_{number:02d}.jpg"))
        for number in event_number.values()
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT), pagesize=landscape(letter), pageCompression=1)
    width, height = landscape(letter)
    page_no = 1

    # Portada y resumen verificable.
    pdf.setFillColor(colors.HexColor("#111827"))
    pdf.rect(0, 0, width, height, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#F6C453"))
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(42, height - 65, "STRIKEVISION ML")
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 30)
    pdf.drawString(42, height - 112, "Auditoria visual de clasificacion")
    pdf.setFont("Helvetica", 17)
    pdf.drawString(42, height - 143, f"Zhang Weili vs Joanna Jedrzejczyk - Round {ROUND_NUMBER}")
    pdf.setFillColor(colors.HexColor("#C7D0DE"))
    pdf.setFont("Helvetica", 10)
    pdf.drawString(42, height - 177, "ID 1 = Zhang (short negro)   |   ID 2 = Joanna (short rojo/borgona)")
    pdf.drawString(42, height - 195, "Cada registro incluye la lamina de cuadros consecutivos del fragmento completo.")
    pdf.drawString(42, height - 211, "La marca SIMULTANEO indica solapamiento temporal con una accion de la rival.")

    counts = Counter(a["label"] for a in actions)
    attackers = Counter(a["attacker"] for a in actions)
    pdf.setFillColor(colors.HexColor("#1F2937"))
    pdf.roundRect(42, 75, width - 84, 255, 12, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(62, 302, f"Resumen: {len(actions)} acciones | {EXPECTED_FRAGMENTS}/{EXPECTED_FRAGMENTS} fragmentos | {len(data.get('source_corrections', []))} correcciones no_strike")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(62, 279, f"Zhang / ID 1: {attackers['1']} acciones")
    pdf.drawString(245, 279, f"Joanna / ID 2: {attackers['2']} acciones")
    pdf.drawString(440, 279, f"Acciones en solapamiento: {len(simultaneous)}")
    y = 246
    for label in ORDER:
        pdf.setFillColor(LABEL_COLORS[label])
        pdf.roundRect(62, y - 4, 13, 13, 3, fill=1, stroke=0)
        pdf.setFillColor(colors.white)
        pdf.setFont("Helvetica", 10)
        pdf.drawString(84, y, f"{LABEL_NAMES[label]}: {counts[label]}")
        y -= 27
    footer(pdf, page_no)
    pdf.showPage()

    cell_w = (width - 70) / 2
    cell_h = (height - 93) / 2
    for label in ORDER:
        items = grouped.get(label, [])
        if not items:
            continue
        for batch_start in range(0, len(items), 4):
            page_no += 1
            color = LABEL_COLORS[label]
            pdf.setFillColor(color)
            pdf.rect(0, height - 52, width, 52, fill=1, stroke=0)
            pdf.setFillColor(colors.white)
            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(28, height - 33, LABEL_NAMES[label])
            pdf.setFont("Helvetica", 9)
            pdf.drawRightString(width - 28, height - 31, f"Clase {label} - {len(items)} registros")

            for offset, action in enumerate(items[batch_start : batch_start + 4]):
                row, col = divmod(offset, 2)
                x = 28 + col * (cell_w + 14)
                y = height - 68 - (row + 1) * cell_h
                pdf.setFillColor(colors.HexColor("#F7F9FC"))
                pdf.setStrokeColor(colors.HexColor("#D7DEE9"))
                pdf.roundRect(x, y, cell_w, cell_h - 8, 7, fill=1, stroke=1)

                attacker = "Zhang / ID 1" if action["attacker"] == "1" else "Joanna / ID 2"
                fragment = event_number[action["event_id"]]
                simult = "  |  SIMULTANEO" if action["action_id"] in simultaneous else ""
                header = f"Fragmento {fragment:02d}  |  {attacker}{simult}"
                pdf.setFillColor(color)
                pdf.setFont("Helvetica-Bold", 9)
                pdf.drawString(x + 10, y + cell_h - 29, header)
                pdf.setFillColor(colors.HexColor("#374151"))
                detail = f"Cuadros {action['start_frame']}-{action['end_frame']}  |  accion {action['action_id'][:8]}"
                fit_text(pdf, detail, cell_w - 20, 7.5)
                pdf.drawString(x + 10, y + cell_h - 43, detail)

                image_y = y + 12
                image_h = cell_h - 69
                draw_frame(
                    pdf,
                    event_sheets[fragment],
                    x + 8,
                    image_y,
                    cell_w - 16,
                    image_h,
                    f"secuencia completa del fragmento {fragment:02d}",
                )
            footer(pdf, page_no)
            pdf.showPage()

    pdf.save()
    print(f"output={OUTPUT}")
    print(f"pages={page_no} actions={len(actions)} sheets={len(actions)} simultaneous_actions={len(simultaneous)}")
    print("counts=" + json.dumps(dict(counts), sort_keys=True))


if __name__ == "__main__":
    main()
