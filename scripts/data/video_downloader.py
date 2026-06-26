from pathlib import Path
import argparse
import re
import shutil
import sys

try:
    import yt_dlp
except ImportError:
    print("Falta instalar yt-dlp. Ejecuta: pip install yt-dlp")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "full_fights"
FFMPEG_SEARCH_PATHS = (
    Path.home() / "AppData/Local/Microsoft/WinGet/Packages",
    Path("C:/ffmpeg/bin"),
    Path("C:/Program Files/ffmpeg/bin"),
)

FIGHT_VIDEOS: dict[str, dict[str, str]] = {
    "adesanya_pereira_1": {
        "url": "https://www.youtube.com/watch?v=VhJlkRKFT9Y&t=1887s",
        "fight": "Israel Adesanya vs Alex Pereira 1",
        "category": "pelea_agresiva_hombres",
    },
    "holloway_gaethje": {
        "url": "https://www.youtube.com/watch?v=DdgEbBpPD_4&t=1649s",
        "fight": "Holloway vs Gaethje",
        "category": "pelea_agresiva_hombres",
    },
    "topuria_holloway": {
        "url": "https://www.youtube.com/watch?v=nGcK0ilhu9c&t=147s",
        "fight": "Ilia Topuria vs Max Holloway",
        "category": "pelea_agresiva_hombres",
    },
    "zhang_joanna_1": {
        "url": "https://www.youtube.com/watch?v=i_AL3LLUnHY",
        "fight": "Zhang Weili vs Joanna Jędrzejczyk 1",
        "category": "pelea_agresiva_mujeres",
    },
    "fiziev_bahamondes": {
        "url": "https://www.youtube.com/watch?v=fMgVtMxusfw",
        "fight": "Rafael Fiziev vs Ignacio Bahamondes",
        "category": "pelea_normal_hombres",
    },
    "yan_sandhagen": {
        "url": "https://www.youtube.com/watch?v=CNfdnJSvoTo",
        "fight": "Petr Yan vs Cory Sandhagen",
        "category": "pelea_normal_hombres",
    },
    # Nueva pelea agregada
    "yan_sandhagen_grappling": {
        "url": "https://www.youtube.com/watch?v=CNfdnJSvoTo&t=26s",
        "fight": "Petr Yan vs Cory Sandhagen",
        "category": "pelea_grappling_leve_hombres",
    },
    "aldo_font": {
        "url": "https://www.youtube.com/watch?v=F3Zi_Lw0S78",
        "fight": "Jose Aldo vs Rob Font",
        "category": "pelea_normal_hombres",
    },
    "fiziev_green": {
        "url": "https://www.youtube.com/watch?v=uVyDxSDdDJ0",
        "fight": "Rafael Fiziev vs Bobby Green",
        "category": "pelea_lowkicks_hombres",
    },
    "romero_adesanya": {
        "url": "https://www.youtube.com/watch?v=ihICs4IbClk",
        "fight": "Yoel Romero vs Israel Adesanya",
        "category": "pelea_pasiva_hombres",
    },
    "joanna_andrade": {
        "url": "https://www.youtube.com/watch?v=SHqs3BccCIk",
        "fight": "Joanna Jędrzejczyk vs Jéssica Andrade",
        "category": "pelea_lowkicks_mujeres",
    },
    "nunes_shevchenko_2": {
        "url": "https://www.youtube.com/watch?v=HvSO0kfcv90",
        "fight": "Nunes vs Shevchenko 2",
        "category": "pelea_masiva_mujeres",
    },
}


def build_output_filename(video_id: str, fight: str, category: str) -> str:
    fight_slug = re.sub(r"[^\w\-]+", "_", fight.lower()).strip("_")
    return f"{video_id}__{fight_slug}__{category}"


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


def build_ytdlp_options(output_template: str) -> dict:
    ffmpeg = find_ffmpeg()
    options: dict = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "nooverwrites": True,
        "remote_components": ["ejs:github"],
        "extractor_args": {
            "youtube": {
                "player_client": ["default", "web_safari", "android", "ios"],
            }
        },
    }

    if ffmpeg:
        options["ffmpeg_location"] = str(ffmpeg.parent)
        return options

    print(
        "Aviso: ffmpeg no encontrado. Instala el binario real con:\n"
        "  winget install Gyan.FFmpeg\n"
        "(pip install ffmpeg NO instala ffmpeg). Se usara formato sin fusionar."
    )
    options["format"] = "best[ext=mp4]/best"
    options.pop("merge_output_format", None)
    return options


def download_youtube_video(
    url: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    filename: str | None = None,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    output_template = (
        str(destination / f"{filename}.%(ext)s")
        if filename
        else str(destination / "%(title)s.%(ext)s")
    )

    options = build_ytdlp_options(output_template)

    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([url])

    return destination / f"{filename}.mp4" if filename else destination


def download_fight(video_id: str, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    if video_id not in FIGHT_VIDEOS:
        available = ", ".join(sorted(FIGHT_VIDEOS))
        raise ValueError(f"ID desconocido: {video_id}. Disponibles: {available}")

    video = FIGHT_VIDEOS[video_id]
    filename = build_output_filename(video_id, video["fight"], video["category"])
    print(f"Descargando [{video_id}] {video['fight']} ({video['category']})...")
    download_youtube_video(video["url"], output_dir, filename)
    print(f"Guardado como: {filename}.mp4")


def download_all_fights(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    total = len(FIGHT_VIDEOS)
    failed: list[str] = []

    for index, video_id in enumerate(FIGHT_VIDEOS, start=1):
        print(f"\n[{index}/{total}]")
        try:
            download_fight(video_id, output_dir)
        except Exception as error:
            print(f"Error en [{video_id}]: {error}")
            failed.append(video_id)

    if failed:
        print(f"\nFallaron {len(failed)} descargas: {', '.join(failed)}")
        sys.exit(1)


def list_fights() -> None:
    for video_id, video in FIGHT_VIDEOS.items():
        print(f"{video_id}: {video['fight']} | {video['category']}")
        print(f"  {video['url']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga videos de peleas UFC desde YouTube. "
        "Sin argumentos descarga todo el catalogo."
    )
    parser.add_argument("url", nargs="?", help="URL individual de YouTube")
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde se guardaran los videos",
    )
    parser.add_argument(
        "--id",
        dest="fight_id",
        help="Descarga una pelea del catalogo por su ID",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Muestra el catalogo de peleas disponibles",
    )
    args = parser.parse_args()

    if args.list:
        list_fights()
        return

    if args.fight_id:
        download_fight(args.fight_id, args.output)
        print("Descarga completada.")
        return

    if args.url:
        download_youtube_video(args.url, args.output)
        print("Descarga completada.")
        return

    download_all_fights(args.output)
    print("\nDescarga de catalogo completada.")


if __name__ == "__main__":
    main()
