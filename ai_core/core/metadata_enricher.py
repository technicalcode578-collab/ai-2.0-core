import musicbrainzngs as mb
from typing import Optional, Dict, Any

# --- Configuration ---
# Set a user-agent, which is required by the MusicBrainz API
mb.set_useragent("Acytel AI", "1.0", "https://github.com/technicalcode578-collab/ai-2-0-core")

def enrich_metadata(artist: str, title: str) -> Optional[Dict[str, Any]]:
    """
    Enriches song metadata by searching the MusicBrainz database.

    Args:
        artist: The name of the song's artist.
        title: The title of the song.

    Returns:
        A dictionary containing enriched metadata (like year and tags),
        or None if no definitive match is found.
    """
    try:
        # Search for recordings that match the artist and title
        result = mb.search_recordings(artist=artist, recording=title, limit=1)
        
        # Check if we found a confident match
        if result['recording-list'] and int(result['recording-list'][0]['ext:score']) > 90:
            recording = result['recording-list'][0]
            
            enriched_data = {}
            
            # Extract the release year
            if 'first-release-date' in recording:
                enriched_data['year'] = recording['first-release-date'].split('-')[0]
            
            # Extract official genre tags
            if 'tag-list' in recording:
                enriched_data['tags'] = [tag['name'] for tag in recording['tag-list']]

            return enriched_data
            
        return None
    except mb.WebServiceError as exc:
        print(f"ERROR: Could not connect to MusicBrainz for '{title}': {exc}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during metadata enrichment for '{title}': {e}")
        return None

