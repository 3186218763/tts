from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dialogue.audio_player import AudioPlayer


@pytest.mark.asyncio
async def test_play_wav_bytes_calls_sounddevice():
    fake_audio = np.zeros(1000, dtype="float32")

    with patch("dialogue.audio_player.sf") as mock_sf, \
         patch("dialogue.audio_player.sd") as mock_sd:
        mock_sf.read.return_value = (fake_audio, 32000)

        player = AudioPlayer()
        await player.play_wav_bytes(b"fake_wav_bytes")

        mock_sf.read.assert_called_once()
        mock_sd.play.assert_called_once_with(fake_audio, 32000)
        mock_sd.wait.assert_called_once()
