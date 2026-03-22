import os
import logging

from supabase import create_client

logger = logging.getLogger(__name__)

# Initialise the Supabase client once at module level.
_supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)


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
    url = res["signedURL"]

    logger.info("Generated signed URL (expires in %ds)", expiry)
    return url
