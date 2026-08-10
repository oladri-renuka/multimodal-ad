"""Configuration system for multimodal content safety reviewer."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import os
import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    raw_dir: Path = Path("./data/raw")
    processed_dir: Path = Path("./data/processed")
    test_dir: Path = Path("./data/test_clips")

    # Dataset splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Classes
    classes: list = field(default_factory=lambda: ["weapon", "nsfw", "counterfeit"])

    # Frame extraction
    target_fps: int = 16
    min_clip_duration: float = 2.0  # seconds
    max_clip_duration: float = 300.0  # 5 minutes

    def __post_init__(self):
        self.raw_dir = Path(self.raw_dir)
        self.processed_dir = Path(self.processed_dir)
        self.test_dir = Path(self.test_dir)


@dataclass
class DetectorConfig:
    """YOLOv8 detector configuration."""
    model_size: str = "n"  # nano
    pretrained_weights: str = "yolov8n.pt"
    checkpoint_path: Optional[Path] = Path("./models/yolov8n_finetuned.pt")

    # Training hyperparams
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    momentum: float = 0.937
    warmup_epochs: int = 3
    early_stopping_patience: int = 15

    # Inference
    conf_threshold: float = 0.45
    iou_threshold: float = 0.5
    max_det: int = 300

    # Device
    device: str = "cuda"
    half: bool = True  # FP16

    def __post_init__(self):
        if self.checkpoint_path:
            self.checkpoint_path = Path(self.checkpoint_path)


@dataclass
class OCRConfig:
    """EasyOCR configuration."""
    languages: list = field(default_factory=lambda: ["en"])
    gpu: bool = True
    conf_threshold: float = 0.3

    # Preprocessing
    padding: int = 10
    max_text_region_area: int = 1000000  # pixels


@dataclass
class ASRConfig:
    """Whisper ASR configuration."""
    model_size: str = "small"  # small, base, medium
    device: str = "cuda"
    batch_size: int = 4
    chunk_length: int = 30  # seconds

    # Language detection
    language: Optional[str] = None  # None = auto-detect


@dataclass
class VLMConfig:
    """Vision Language Model configuration."""
    model_name: str = "liuhaotian/llava-v1.5-7b"
    device: str = "cuda"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Quantization
    nf4: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"

    # Model id cache
    checkpoint_path: Optional[Path] = Path("./models/llava_quantized.pt")

    def __post_init__(self):
        if self.checkpoint_path:
            self.checkpoint_path = Path(self.checkpoint_path)


@dataclass
class MetricsConfig:
    """Metrics and evaluation configuration."""
    output_dir: Path = Path("./metrics_output")
    confusion_matrix_figsize: tuple = (10, 8)
    compute_per_class_metrics: bool = True

    # Ablation study
    run_ablation: bool = True
    ablation_configs: list = field(default_factory=lambda: [
        "detector_only",
        "detector_ocr",
        "detector_vlm",
        "full_pipeline"
    ])

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 2
    reload: bool = False

    # Upload settings
    max_video_size_mb: int = 500
    temp_upload_dir: Path = Path("./temp/uploads")
    allowed_formats: list = field(default_factory=lambda: [
        "mp4", "avi", "mov", "mkv", "flv", "wmv"
    ])

    # Async processing
    max_queue_size: int = 10
    job_timeout_seconds: int = 600

    def __post_init__(self):
        self.temp_upload_dir = Path(self.temp_upload_dir)


@dataclass
class Config:
    """Master configuration."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # Logging
    log_level: str = "INFO"
    log_dir: Path = Path("./logs")

    def __post_init__(self):
        self.log_dir = Path(self.log_dir)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        config_dict = asdict(self)
        # Convert Path objects to strings and tuples to lists for YAML serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        return convert_types(config_dict)

    def to_yaml(self, filepath: Path) -> None:
        """Save configuration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filepath: Path) -> "Config":
        """Load configuration from YAML file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Reconstruct nested dataclasses
        return cls(
            dataset=DatasetConfig(**config_dict.get("dataset", {})),
            detector=DetectorConfig(**config_dict.get("detector", {})),
            ocr=OCRConfig(**config_dict.get("ocr", {})),
            asr=ASRConfig(**config_dict.get("asr", {})),
            vlm=VLMConfig(**config_dict.get("vlm", {})),
            metrics=MetricsConfig(**config_dict.get("metrics", {})),
            api=APIConfig(**config_dict.get("api", {})),
            log_level=config_dict.get("log_level", "INFO"),
            log_dir=Path(config_dict.get("log_dir", "./logs"))
        )

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()

        # Override from env vars
        if os.getenv("DATA_RAW_DIR"):
            config.dataset.raw_dir = Path(os.getenv("DATA_RAW_DIR"))
        if os.getenv("DETECTOR_CHECKPOINT"):
            config.detector.checkpoint_path = Path(os.getenv("DETECTOR_CHECKPOINT"))
        if os.getenv("DETECTOR_CONF_THRESHOLD"):
            config.detector.conf_threshold = float(os.getenv("DETECTOR_CONF_THRESHOLD"))
        if os.getenv("DEVICE"):
            config.detector.device = os.getenv("DEVICE")
            config.vlm.device = os.getenv("DEVICE")
            config.ocr.gpu = os.getenv("DEVICE") == "cuda"

        return config


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
