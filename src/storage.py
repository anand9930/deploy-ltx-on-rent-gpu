import os
import logging

from supabase import create_client

logger = logging.getLogger(__name__)

# Validate required env vars early with a clear error.
_url = os.getenv("SUPABASE_URL")
_key = os.getenv("SUPABASE_SERVICE_KEY")
if not _url or not _key:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables must be set"
    )

# Initialise the Supabase client once at module level.
_supabase = create_client(_url, _key)


def upload_video(file_path: str, object_key: str) -> str:
    """Upload an MP4 file to Supabase Storage and return a signed URL."""
    bucket = os.getenv("SUPABASE_BUCKET", "ltx-videos")

    logger.info("Uploading %s to supabase://%s/%s", file_path, bucket, object_key)

    with open(file_path, "rb") as f:
        _supabase.storage.from_(bucket).upload(
            path=object_key,
            file=f,
            file_options={"content-type": "video/mp4"},
        )

    expiry = int(os.getenv("SUPABASE_URL_EXPIRY_SECONDS", "604800"))
    res = _supabase.storage.from_(bucket).create_signed_url(
        path=object_key,
        expires_in=expiry,
    )

    # Handle different SDK response formats
    if isinstance(res, str):
        url = res
    elif isinstance(res, dict):
        url = res.get("signedURL") or res.get("signedUrl", "")
    else:
        url = str(res)

    logger.info("Generated signed URL (expires in %ds)", expiry)
    return url
