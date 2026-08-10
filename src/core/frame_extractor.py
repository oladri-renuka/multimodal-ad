"""Video frame extraction utility."""

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2
import ffmpeg

logger = logging.getLogger(__name__)


class VideoMetadata:
    """Video metadata container."""
    def __init__(self, width: int, height: int, fps: float, total_frames: int, duration: float):
        self.width = width
        self.height = height
        self.fps = fps
        self.total_frames = total_frames
        self.duration = duration

    def __repr__(self):
        return (f"VideoMetadata(width={self.width}, height={self.height}, "
                f"fps={self.fps}, frames={self.total_frames}, duration={self.duration:.2f}s)")


class FrameExtractor:
    """Extract frames and audio from video files."""

    @staticmethod
    def get_metadata(video_path: str) -> VideoMetadata:
        """Extract video metadata using ffmpeg."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        try:
            probe = ffmpeg.probe(str(video_path))
            video_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "video"),
                None
            )

            if not video_stream:
                raise ValueError(f"No video stream found in {video_path}")

            width = int(video_stream["width"])
            height = int(video_stream["height"])
            fps = eval(video_stream["r_frame_rate"])
            duration = float(probe["format"]["duration"])
            total_frames = int(duration * fps)

            return VideoMetadata(width, height, fps, total_frames, duration)

        except Exception as e:
            logger.error(f"Error probing video {video_path}: {e}")
            raise

    @staticmethod
    def extract_frames(
        video_path: str,
        output_dir: str,
        target_fps: int = 16,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        frame_format: str = "frame_%06d.jpg"
    ) -> Tuple[list, VideoMetadata]:
        """
        Extract frames from video at target FPS.

        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            target_fps: Target frames per second (default 16)
            start_sec: Start time in seconds
            end_sec: End time in seconds (None = full video)
            frame_format: Frame filename pattern

        Returns:
            Tuple of (list of frame paths, VideoMetadata)
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        metadata = FrameExtractor.get_metadata(str(video_path))

        # Calculate output FPS ratio
        scale_factor = target_fps / metadata.fps

        try:
            # Build ffmpeg command
            stream = ffmpeg.input(str(video_path))

            # Add trim filter if needed
            filter_chain = [f"fps={target_fps}"]
            if start_sec > 0 or end_sec is not None:
                end = end_sec if end_sec else metadata.duration
                filter_chain.insert(0, f"trim=start={start_sec}:end={end}")

            stream = ffmpeg.filter(stream, "fps", fps=target_fps)

            stream = ffmpeg.output(
                stream,
                str(output_dir / frame_format),
                video_bitrate="0",  # No re-encoding, just frame extraction
                q=2  # Quality (1=best, 31=worst for JPEG)
            )

            # Run extraction
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)

            # Collect frame paths
            frame_paths = sorted(output_dir.glob("frame_*.jpg"))

            if not frame_paths:
                raise ValueError(f"No frames extracted from {video_path}")

            logger.info(f"Extracted {len(frame_paths)} frames from {video_path}")
            return frame_paths, metadata

        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise

    @staticmethod
    def extract_audio(
        video_path: str,
        output_path: str,
        audio_format: str = "wav",
        sample_rate: int = 16000
    ) -> str:
        """
        Extract audio from video.

        Args:
            video_path: Path to video file
            output_path: Path to save audio
            audio_format: Audio format (wav, mp3, etc.)
            sample_rate: Sample rate in Hz

        Returns:
            Path to extracted audio file
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        try:
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(
                stream["a"],
                str(output_path),
                acodec="pcm_s16le" if audio_format == "wav" else "libmp3lame",
                ar=sample_rate
            )

            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)

            logger.info(f"Extracted audio to {output_path} (sample_rate={sample_rate}Hz)")
            return str(output_path)

        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise

    @staticmethod
    def read_frame(frame_path: str) -> np.ndarray:
        """
        Read a single frame from disk.

        Args:
            frame_path: Path to frame image

        Returns:
            Frame as numpy array (BGR format)
        """
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")
        return frame

    @staticmethod
    def read_frames_batch(
        frame_paths: list,
        max_batch_size: int = 32
    ) -> list:
        """
        Read multiple frames in batches.

        Args:
            frame_paths: List of frame paths
            max_batch_size: Maximum frames per batch

        Returns:
            List of numpy arrays
        """
        frames = []
        for i, frame_path in enumerate(frame_paths):
            try:
                frame = FrameExtractor.read_frame(frame_path)
                frames.append(frame)
            except Exception as e:
                logger.warning(f"Failed to read frame {i}: {e}")
                continue

        return frames

    @staticmethod
    def validate_video(video_path: str) -> bool:
        """
        Validate video file integrity.

        Args:
            video_path: Path to video file

        Returns:
            True if valid, False otherwise
        """
        try:
            metadata = FrameExtractor.get_metadata(video_path)
            return metadata.total_frames > 0 and metadata.duration > 0
        except Exception as e:
            logger.error(f"Video validation failed: {e}")
            return False
