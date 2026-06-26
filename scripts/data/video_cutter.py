from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "full_fights"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_ROUNDS_FILE = PROJECT_ROOT / "data" / "metadata" / "rounds.json"
FFMPEG_SEARCH_PATHS = (
    Path.home() / "AppData/Local/Microsoft/WinGet/Packages",
    Path("C:/ffmpeg/bin"),
    Path("C:/Program Files/ffmpeg/bin"),
)


def find_ffmpeg() -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return Path(ffmpeg)

    for search_path in FFMPEG_SEARCH_PATHS:
        if not search_path.exists():
            continue
        for candidate in search_path.glob("**/ffmpeg.exe"):
            return candidate

    return None


def timestamp_to_seconds(timestamp: str) -> float:
    parts = timestamp.strip().strip(":").split(":")
    if not all(part.strip() != "" for part in parts):
        raise ValueError(f"Timestamp invalido: {timestamp!r}")

    numbers = [float(part) for part in parts]
    seconds = 0.0
    for value in numbers:
        seconds = seconds * 60 + value
    return seconds


def find_source_file(source_dir: Path, source_prefix: str) -> Path | None:
    prefix = source_prefix.lower()
    for candidate in source_dir.glob("*.mp4"):
        if candidate.name.lower().startswith(prefix):
            return candidate
    return None


def cut_round(
    ffmpeg: Path,
    source: Path,
    output_path: Path,
    start: float,
    duration: float,
    reencode: bool,
) -> None:
    command = [
        str(ffmpeg),
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
    ]

    if reencode:
        command += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
        ]
    else:
        command += ["-c", "copy", "-avoid_negative_ts", "make_zero"]

    command.append(str(output_path))

    subprocess.run(command, check=True)


def process_fights(
    fights: list[dict],
    source_dir: Path,
    output_dir: Path,
    ffmpeg: Path,
    reencode: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    failed: list[str] = []
    created = 0

    for fight in fights:
        fight_id = fight["id"]
        source = find_source_file(source_dir, fight["source_prefix"])
        if source is None:
            print(f"[SKIP] No se encontro el video para '{fight_id}' "
                  f"(prefijo: {fight['source_prefix']})")
            missing.append(fight_id)
            continue

        category = fight["output_base"].split("__")[-1]
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {fight_id} -> {source.name} (categoria: {category}) ===")
        for round_info in fight["rounds"]:
            number = round_info["round"]
            start = timestamp_to_seconds(round_info["start"])
            end = timestamp_to_seconds(round_info["end"])
            duration = end - start

            if duration <= 0:
                print(f"  [ERROR] Round {number}: fin <= inicio, se omite.")
                failed.append(f"{fight_id}_round{number}")
                continue

            output_path = category_dir / f"{fight['output_base']}_round{number}.mp4"
            print(f"  Round {number}: {round_info['start']} -> {round_info['end']} "
                  f"({duration:.0f}s) -> {output_path.name}")

            try:
                cut_round(ffmpeg, source, output_path, start, duration, reencode)
                created += 1
            except subprocess.CalledProcessError as error:
                print(f"  [ERROR] ffmpeg fallo en round {number}: {error}")
                failed.append(f"{fight_id}_round{number}")

    print(f"\nClips creados: {created}")
    if missing:
        print(f"Videos no encontrados: {', '.join(missing)}")
    if failed:
        print(f"Rounds fallidos: {', '.join(failed)}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recorta cada round de los videos de pelea segun rounds.json."
    )
    parser.add_argument(
        "--rounds",
        type=Path,
        default=DEFAULT_ROUNDS_FILE,
        help="Ruta al JSON con los timestamps de cada round",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Carpeta con los videos completos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde se guardaran los clips por round",
    )
    parser.add_argument(
        "--id",
        dest="fight_id",
        help="Procesa solo una pelea por su id del JSON",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Reencodea para cortes exactos (mas lento). Por defecto copia streams.",
    )
    args = parser.parse_args()

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print("ffmpeg no encontrado. Instalalo con: winget install Gyan.FFmpeg")
        sys.exit(1)

    if not args.rounds.exists():
        print(f"No existe el archivo de rounds: {args.rounds}")
        sys.exit(1)

    if not args.source.exists():
        print(f"No existe la carpeta de videos: {args.source}")
        sys.exit(1)

    data = json.loads(args.rounds.read_text(encoding="utf-8"))
    fights = data["fights"]

    if args.fight_id:
        fights = [fight for fight in fights if fight["id"] == args.fight_id]
        if not fights:
            print(f"No se encontro la pelea con id '{args.fight_id}' en el JSON.")
            sys.exit(1)

    process_fights(fights, args.source, args.output, ffmpeg, args.reencode)


if __name__ == "__main__":
    main()
