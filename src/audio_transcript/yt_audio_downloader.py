"""
Downloads audio from a YouTube URL using yt-dlp.

yt-dlp must be installed from the nightly build (not PyPI) to stay in sync with YouTube API changes:

    python3 -m pip install --force-reinstall https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz
"""

import os
from typing import Any, Dict
import yt_dlp
from pydub.utils import mediainfo

from .utils import rename_file


def download_audio(url: str, file_title: str, output_folder: str = "audio_files") -> Dict[str, Any]:
    filepath = f"{output_folder}/{file_title}"

    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with yt_dlp.YoutubeDL({'extract_audio': True, 'format': 'bestaudio', 'outtmpl': filepath}) as ydl:
            ydl.download(url)

        response: Dict[str, Any] = {
            "status": True,
            "message": f"Successfully downloaded audio file from {url}",
        }

        audio_type = mediainfo(filepath=filepath)['codec_name']
        new_filepath = f"{output_folder}/{file_title}.{audio_type}"

        rename_response = rename_file(old_filepath=filepath, new_filepath=new_filepath)
        if rename_response['status']:
            response["output_file"] = rename_response["filepath"]
        else:
            response["message"] = (
                f"{response['message']}\nFile saved but rename failed: {rename_response['message']}."
                f"\nPlease add the extension manually to {filepath}"
            )
            response["output_file"] = filepath

        return response

    except Exception as e:
        return {"status": False, "message": f"Failed to download audio file: {e}"}


def main(yt_url: str, file_title: str, debug: bool = False) -> Dict[str, Any]:
    response = download_audio(url=yt_url, file_title=file_title)
    if debug:
        print(f"\nyt_audio_downloader.py:\n{response}\n")
    return response
