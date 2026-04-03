"""Shared test fixtures — mock GPU pipeline for local testing."""

import os
import tempfile
import uuid

import pytest


class MockGenerator:
    """Drop-in replacement for LTXVideoGenerator that runs without GPU.

    Returns a tiny valid MP4 file so the full serving flow
    (BentoML endpoint → generate → encode → response) can be tested locally.
    """

    def generate(self, **kwargs):
        job_id = uuid.uuid4().hex[:12]
        output_filename = f"ltx_{job_id}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        # Minimal valid MP4 (ftyp + moov atoms)
        _write_dummy_mp4(output_path)

        return {
            "output_path": output_path,
            "output_filename": output_filename,
            "generation_time_seconds": 0.01,
            "parameters": {
                "width": kwargs.get("width", 512),
                "height": kwargs.get("height", 768),
                "num_frames": kwargs.get("num_frames", 25),
                "num_inference_steps": kwargs.get("num_inference_steps", 8),
                "seed": kwargs.get("seed", 42),
                "frame_rate": kwargs.get("frame_rate", 24.0),
                "cfg_scale": kwargs.get("cfg_scale", 3.0),
                "stg_scale": kwargs.get("stg_scale", 1.0),
                "rescale_scale": kwargs.get("rescale_scale", 0.7),
            },
        }


def _write_dummy_mp4(path: str) -> None:
    """Write a minimal valid MP4 file (~24 bytes)."""
    import struct

    ftyp = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42"
    with open(path, "wb") as f:
        f.write(ftyp)


@pytest.fixture
def mock_generator():
    return MockGenerator()
