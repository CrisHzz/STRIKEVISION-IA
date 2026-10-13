"""Human action annotations over merged, previously reviewed strike intervals."""

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

LABELS = {"7": "punch_head", "9": "punch_body", "4": "kick_leg", "5": "kick_body",
          "6": "kick_head", "1": "other_strike", "3": "ambiguous"}
LOCK = threading.RLock()


def strike_intervals(rows):
    """Union only positive windows; retain original IDs and absolute frame bounds."""
    output = []
    for row in sorted((r for r in rows if r.get("label") == "strike"), key=lambda r: r["start_frame"]):
        start, end = int(row["start_frame"]), int(row["end_frame"])
        if output and start <= output[-1]["end_frame"] + 1:
            output[-1]["end_frame"] = max(end, output[-1]["end_frame"])
            output[-1]["window_ids"].append(row["window_id"])
        else:
            output.append({"start_frame": start, "end_frame": end, "window_ids": [row["window_id"]]})
    for row in output:
        row["event_id"] = f"f{row['start_frame']}-{row['end_frame']}"
    return output


class ActionStore:
    """Atomic storage with optimistic revisions to reject duplicate/stale clicks."""

    def __init__(self, source: Path, destination: Path):
        self.source, self.path = source, destination
        payload = source.read_bytes()
        self.source_hash = hashlib.sha256(payload).hexdigest()
        rows = [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
        if not rows:
            raise ValueError("Este round no tiene anotaciones.")
        self.video = rows[0]["video_path"]
        self.pose = rows[0]["pose_path"]
        self.video_id = rows[0]["video_id"]
        self.fps = float(rows[0]["fps"])
        if not self.fps > 0:
            raise ValueError("El round debe declarar un FPS válido.")
        self.events = strike_intervals(rows)
        if not self.events:
            raise ValueError("Este round no tiene ventanas etiquetadas como strike.")

    def read(self):
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data["source_sha256"] != self.source_hash:
                raise ValueError("Las anotaciones binarias cambiaron. Revisa la correspondencia antes de continuar.")
            data.setdefault("source_corrections", [])
            return data
        return {"schema_version": "strike_actions_v1", "source_sha256": self.source_hash,
                "video_id": self.video_id, "video_path": self.video, "fps": self.fps,
                "revision": 0, "actions": [], "source_corrections": [], "completed": [], "history": []}

    def commit(self, revision, event_id, *, label=None, attacker="unknown", start=None, end=None,
               complete=False, undo=False, source_correction=False):
        with LOCK:
            data = self.read()
            if data["revision"] != revision:
                raise ValueError("Pulsación repetida o vista desactualizada. Recarga el round; la última anotación está guardada.")
            if undo:
                if not data["history"]:
                    raise ValueError("No hay decisiones para deshacer.")
                previous = data["history"].pop()
                event_id = previous["event_id"]
                data["actions"] = data["actions"][:previous["action_count"]]
                data["source_corrections"] = data["source_corrections"][:previous.get("correction_count", 0)]
                if not previous["was_completed"] and event_id in data["completed"]:
                    data["completed"].remove(event_id)
            else:
                event = next(e for e in self.events if e["event_id"] == event_id)
                if event_id in data['completed']:
                    raise ValueError('Este fragmento ya está terminado. Usa deshacer para corregirlo.')
                if source_correction and any(a["event_id"] == event_id for a in data["actions"]):
                    raise ValueError("Deshaz las acciones de este fragmento antes de marcarlo como no_strike.")
                if label is None and complete and not any(a["event_id"] == event_id for a in data["actions"]):
                    if not source_correction:
                        raise ValueError("Anota al menos una acción antes de terminar el fragmento.")
                if label is not None:
                    if label not in LABELS.values() or attacker not in ("1", "2", "unknown"):
                        raise ValueError("Categoría o atacante inválido.")
                    if start is None or end is None or int(start) != start or int(end) != end:
                        raise ValueError("Los límites deben ser frames enteros.")
                    if not event["start_frame"] <= start <= end <= event["end_frame"]:
                        raise ValueError("La acción debe estar dentro del fragmento de strike.")
                    if any(a["event_id"] == event_id and a["start_frame"] == start and a["end_frame"] == end
                           and a["attacker"] == attacker and a["label"] == label for a in data["actions"]):
                        raise ValueError("Esta acción ya está guardada.")
                data["history"].append({"action_count": len(data["actions"]),
                                        "correction_count": len(data["source_corrections"]),
                                        "was_completed": event_id in data["completed"], "event_id": event_id})
                if label is not None:
                    data["actions"].append({"action_id": uuid.uuid4().hex, "event_id": event_id,
                        "source_window_ids": event["window_ids"], "start_frame": int(start), "end_frame": int(end),
                        "attacker": attacker, "label": label, "annotated_at": datetime.now(timezone.utc).isoformat()})
                if source_correction:
                    data["source_corrections"].append({
                        "correction_id": uuid.uuid4().hex,
                        "event_id": event_id,
                        "source_window_ids": event["window_ids"],
                        "start_frame": event["start_frame"],
                        "end_frame": event["end_frame"],
                        "label": "no_strike",
                        "reason": "prior_strike_annotation_error",
                        "annotated_at": datetime.now(timezone.utc).isoformat(),
                    })
                if complete and event_id not in data["completed"]:
                    data["completed"].append(event_id)
            data["revision"] += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
            return data, event_id
