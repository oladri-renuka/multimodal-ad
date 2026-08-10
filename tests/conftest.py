"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import logging
from pathlib import Path
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary test data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_video_file(test_data_dir):
    """Create a sample video file for testing."""
    video_path = test_data_dir / "sample.mp4"

    # Create a simple video with frames
    width, height = 640, 480
    fps = 30
    duration = 2  # seconds
    total_frames = fps * duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        pytest.skip("Could not create test video")

    # Generate simple test frames (alternating colors)
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Create a simple pattern
        color = (0, 255 * (i % 2), 255)
        frame[:] = color

        # Add text
        cv2.putText(
            frame,
            f"Frame {i+1}/{total_frames}",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        writer.write(frame)

    writer.release()

    yield video_path

    # Cleanup
    if video_path.exists():
        video_path.unlink()


@pytest.fixture
def sample_audio_file(test_data_dir):
    """Create a sample audio file for testing."""
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")

    audio_path = test_data_dir / "sample.wav"

    # Create simple sine wave
    sample_rate = 16000
    duration = 2  # seconds
    frequency = 440  # Hz (A4 note)

    t = np.arange(int(sample_rate * duration)) / sample_rate
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)

    sf.write(str(audio_path), audio, sample_rate)

    yield audio_path

    # Cleanup
    if audio_path.exists():
        audio_path.unlink()


@pytest.fixture
def sample_frames_dir(test_data_dir):
    """Create a directory with sample frames."""
    frames_dir = test_data_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Create 10 sample frames
    for i in range(10):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        frame_path = frames_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)

    yield frames_dir

    # Cleanup
    for frame_path in frames_dir.glob("*.jpg"):
        frame_path.unlink()


@pytest.fixture
def config_object():
    """Create a test configuration object."""
    from src.core.config import Config, DatasetConfig

    config = Config()
    config.dataset = DatasetConfig()

    return config


@pytest.fixture
def logger_fixture():
    """Get logger for tests."""
    return logging.getLogger("test")


def pytest_configure(config):
    """Configure pytest."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
