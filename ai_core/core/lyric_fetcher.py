import lyricsgenius
from decouple import config
from typing import Optional

# Fetch the access token from our .env file
GENIUS_ACCESS_TOKEN = config("GENIUS_ACCESS_TOKEN")

# Initialize the Genius API client
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, verbose=False, remove_section_headers=True)

def get_lyrics(artist: str, title: str) -> Optional[str]:
    """
    Fetches lyrics for a given song from the Genius API.

    Args:
        artist: The name of the song's artist.
        title: The title of the song.

    Returns:
        The lyrics as a string, or None if not found or an error occurs.
    """
    try:
        song = genius.search_song(title, artist)
        if song and song.lyrics:
            # Clean up the lyrics by removing the first line (title) and "Embed" at the end
            lines = song.lyrics.split('\n')
            cleaned_lyrics = '\n'.join(lines[1:])
            # Genius sometimes adds "Embed" at the very end of the string
            if cleaned_lyrics.endswith("Embed"):
                 cleaned_lyrics = cleaned_lyrics[:-5].strip()
            return cleaned_lyrics
        return None
    except Exception as e:
        print(f"An error occurred while fetching lyrics for {title} by {artist}: {e}")
        return None
