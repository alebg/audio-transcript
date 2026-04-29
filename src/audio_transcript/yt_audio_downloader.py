"""
Downloads audio from a YouTube URL using yt-dlp.

yt-dlp must be installed from the nightly build (not PyPI) to stay in sync with YouTube API changes:

    python3 -m pip install --force-reinstall https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz
"""

import os
import yt_dlp
from pydub.utils import mediainfo

from .models import Err, Ok
from .utils import rename_file


def download_audio(url: str, file_title: str, output_folder: str = "audio_files") -> Ok[str] | Err:
    filepath = f"{output_folder}/{file_title}"

    try:
        os.makedirs(output_folder, exist_ok=True)

        with yt_dlp.YoutubeDL({'extract_audio': True, 'format': 'bestaudio', 'outtmpl': filepath}) as ydl:
            ydl.download(url)

        audio_type = mediainfo(filepath=filepath)['codec_name']
        new_filepath = f"{output_folder}/{file_title}.{audio_type}"

        rename_result = rename_file(old_filepath=filepath, new_filepath=new_filepath)
        if isinstance(rename_result, Err):
            return Err(message=f"Downloaded to {filepath} but rename failed: {rename_result.message}")

        return Ok(value=rename_result.value)

    except Exception as e:
        return Err(message=f"Failed to download audio file: {e}")


def main(yt_url: str, file_title: str, debug: bool = False) -> Ok[str] | Err:
    result = download_audio(url=yt_url, file_title=file_title)
    if debug:
        print(f"\nyt_audio_downloader.py:\n{result}\n")
    return result
