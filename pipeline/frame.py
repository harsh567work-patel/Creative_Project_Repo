
import yt_dlp
from pathlib import Path

# 1. Paste your YouTube playlist URL here
playlist_url = "https://youtube.com/playlist?list=PLbJhGqY-mq47k_WLUtzVjmarUm1EuXPj2&si=YS-150cy8Whzpq7m"

# 2. Create a folder for downloaded audio (points to project data/raw_downloads)
_ROOT = Path(__file__).resolve().parent.parent
download_folder = _ROOT / "data" / "raw_downloads"
download_folder.mkdir(parents=True, exist_ok=True)

# 3. Configure yt-dlp
options = {
    "format": "bestaudio/best",
    "outtmpl": str(
        download_folder / "%(playlist_index)03d - %(title).150s [%(id)s].%(ext)s"
    ),
    "noplaylist": False,
    "ignoreerrors": True
}

# 4. Download audio from the playlist
with yt_dlp.YoutubeDL(options) as ydl:
    ydl.download([playlist_url])

print("\nPlaylist processing finished!")
print("Check the downloads folder:", download_folder)

