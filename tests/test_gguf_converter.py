"""Tests for the GGUF → monolithic safetensors converter."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

# Add src/ to path
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestBuildMetadata:
    def test_model_version_is_2_3(self):
        from gguf_converter import _build_metadata

        meta = _build_metadata()
        assert meta["model_version"] == "2.3"

    def test_config_is_valid_json(self):
        from gguf_converter import _build_metadata

        meta = _build_metadata()
        config = json.loads(meta["config"])
        assert "transformer" in config

    def test_transformer_config_has_v2_flags(self):
        from gguf_converter import _build_metadata

        meta = _build_metadata()
        config = json.loads(meta["config"])
        tf = config["transformer"]
        # V2 feature extractor flags (required for 22B models)
        assert tf["caption_proj_before_connector"] is True
        assert tf["caption_projection_first_linear"] is False
        assert tf["caption_proj_input_norm"] is False
        assert tf["caption_projection_second_linear"] is False
        # Architecture dims used by FeatureExtractorV2
        assert tf["num_attention_heads"] == 32
        assert tf["attention_head_dim"] == 128
        assert tf["audio_num_attention_heads"] == 32
        assert tf["audio_attention_head_dim"] == 64


class TestLoadSafetensorsWithPrefix:
    def test_adds_prefix(self, tmp_path):
        from safetensors.torch import save_file
        from gguf_converter import _load_safetensors_with_prefix

        sd = {"weight": torch.zeros(2, 3), "bias": torch.ones(3)}
        path = str(tmp_path / "test.safetensors")
        save_file(sd, path)

        result = _load_safetensors_with_prefix(path, prefix="vae.")
        assert "vae.weight" in result
        assert "vae.bias" in result
        assert "weight" not in result

    def test_no_prefix(self, tmp_path):
        from safetensors.torch import save_file
        from gguf_converter import _load_safetensors_with_prefix

        sd = {"a": torch.zeros(1)}
        path = str(tmp_path / "test.safetensors")
        save_file(sd, path)

        result = _load_safetensors_with_prefix(path)
        assert "a" in result

    def test_strip_and_add_prefix(self, tmp_path):
        from safetensors.torch import save_file
        from gguf_converter import _load_safetensors_with_prefix

        sd = {"old.weight": torch.zeros(2)}
        path = str(tmp_path / "test.safetensors")
        save_file(sd, path)

        result = _load_safetensors_with_prefix(path, prefix="new.", strip_prefix="old.")
        assert "new.weight" in result


class TestConvertGgufToSafetensors:
    @patch("gguf_converter._load_gguf_as_state_dict")
    def test_full_conversion(self, mock_gguf_load, tmp_path):
        from safetensors.torch import save_file, load_file
        from gguf_converter import convert_gguf_to_safetensors

        mock_gguf_load.return_value = {
            "model.diffusion_model.block.0.weight": torch.randn(4, 4),
        }

        vae_path = str(tmp_path / "video_vae.safetensors")
        save_file({"decoder.conv.weight": torch.randn(3, 3)}, vae_path)

        audio_path = str(tmp_path / "audio_vae.safetensors")
        save_file({"decoder.conv.weight": torch.randn(2, 2)}, audio_path)

        conn_path = str(tmp_path / "connectors.safetensors")
        save_file({
            "text_embedding_projection.weight": torch.randn(5, 5),
            "model.diffusion_model.video_embeddings_connector.weight": torch.randn(3, 3),
        }, conn_path)

        output_path = str(tmp_path / "output.safetensors")

        convert_gguf_to_safetensors(
            gguf_path="dummy.gguf",
            video_vae_path=vae_path,
            audio_vae_path=audio_path,
            connectors_path=conn_path,
            output_path=output_path,
        )

        assert os.path.exists(output_path)

        sd = load_file(output_path)
        assert "model.diffusion_model.block.0.weight" in sd
        assert "vae.decoder.conv.weight" in sd
        assert "audio_vae.decoder.conv.weight" in sd
        assert "text_embedding_projection.weight" in sd
        assert "model.diffusion_model.video_embeddings_connector.weight" in sd

    @patch("gguf_converter._load_gguf_as_state_dict")
    def test_metadata_preserved(self, mock_gguf_load, tmp_path):
        import safetensors
        from safetensors.torch import save_file
        from gguf_converter import convert_gguf_to_safetensors

        mock_gguf_load.return_value = {"model.diffusion_model.w": torch.zeros(1)}

        for name in ("video_vae", "audio_vae", "connectors"):
            save_file({"x": torch.zeros(1)}, str(tmp_path / f"{name}.safetensors"))

        output_path = str(tmp_path / "output.safetensors")
        convert_gguf_to_safetensors(
            gguf_path="dummy.gguf",
            video_vae_path=str(tmp_path / "video_vae.safetensors"),
            audio_vae_path=str(tmp_path / "audio_vae.safetensors"),
            connectors_path=str(tmp_path / "connectors.safetensors"),
            output_path=output_path,
        )

        with safetensors.safe_open(output_path, framework="pt") as f:
            meta = f.metadata()

        assert meta["model_version"] == "2.3"
        config = json.loads(meta["config"])
        assert config["transformer"]["caption_proj_before_connector"] is True


class TestVocoderKeyHandling:
    @patch("gguf_converter._load_gguf_as_state_dict")
    def test_vocoder_keys_remapped(self, mock_gguf_load, tmp_path):
        from safetensors.torch import save_file, load_file
        from gguf_converter import convert_gguf_to_safetensors

        mock_gguf_load.return_value = {"model.diffusion_model.w": torch.zeros(1)}

        vae_path = str(tmp_path / "video_vae.safetensors")
        save_file({"x": torch.zeros(1)}, vae_path)

        audio_path = str(tmp_path / "audio_vae.safetensors")
        save_file({
            "decoder.conv.weight": torch.randn(2, 2),
            "vocoder.conv_pre.weight": torch.randn(3, 3),
        }, audio_path)

        conn_path = str(tmp_path / "connectors.safetensors")
        save_file({"text_embedding_projection.w": torch.zeros(1)}, conn_path)

        output_path = str(tmp_path / "output.safetensors")
        convert_gguf_to_safetensors("d.gguf", vae_path, audio_path, conn_path, output_path)

        sd = load_file(output_path)
        assert "vocoder.conv_pre.weight" in sd
        assert "audio_vae.decoder.conv.weight" in sd


class TestDownloadModels:
    @patch("download_models.hf_hub_download")
    @patch("download_models.snapshot_download")
    @patch.dict(os.environ, {"HF_TOKEN": "test", "USE_GGUF": "0"})
    def test_original_mode_skips_when_cached(self, mock_snap, mock_hf, tmp_path):
        from download_models import _download_original_models

        model_dir = str(tmp_path)
        for f in [
            "ltx-2.3-22b-dev.safetensors",
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "ltx-2.3-22b-distilled-lora-384.safetensors",
        ]:
            (tmp_path / f).touch()
        gemma_dir = tmp_path / "gemma-3-12b-it-qat-q4_0-unquantized"
        gemma_dir.mkdir()
        (gemma_dir / "model.safetensors").touch()

        _download_original_models(model_dir)
        mock_hf.assert_not_called()
        mock_snap.assert_not_called()

    @patch("download_models.hf_hub_download")
    @patch("download_models.snapshot_download")
    @patch.dict(os.environ, {"HF_TOKEN": "test", "USE_GGUF": "1"})
    def test_distilled_mode_skips_when_cached(self, mock_snap, mock_hf, tmp_path):
        from download_models import _download_distilled_models

        model_dir = str(tmp_path)
        (tmp_path / "ltx-2.3-22b-distilled.safetensors").touch()
        (tmp_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()
        gemma_dir = tmp_path / "gemma-3-12b-it-qat-q4_0-unquantized"
        gemma_dir.mkdir()
        (gemma_dir / "model.safetensors").touch()

        _download_distilled_models(model_dir)
        mock_hf.assert_not_called()
        mock_snap.assert_not_called()
