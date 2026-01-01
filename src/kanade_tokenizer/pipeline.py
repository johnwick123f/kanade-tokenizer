from dataclasses import dataclass
from typing import Literal

import jsonargparse
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .data.datamodule import AudioBatch
from .model import KanadeModel, KanadeModelConfig
from .module.audio_feature import MelSpectrogramFeature
from .module.discriminator import SpectrogramDiscriminator
from .module.fsq import FiniteScalarQuantizer
from .module.global_encoder import GlobalEncoder
from .module.postnet import PostNet
from .module.ssl_extractor import SSLFeatureExtractor
from .module.transformer import Transformer
from .util import freeze_modules, get_logger

logger = get_logger()


@dataclass
class KanadePipelineConfig:
    # Training control
    train_feature: bool = True  # Whether to train the feature reconstruction branch
    train_mel: bool = True  # Whether to train the mel spectrogram generation branch

    # Audio settings
    audio_length: int = 138240  # Length of audio input in samples

    # Optimization settings
    lr: float = 2e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    gradient_clip_val: float | None = 1.0

    # LR scheduling parameters
    warmup_percent: float = 0.1
    lr_div_factor: float = 10.0
    lr_final_div_factor: float = 1.0
    anneal_mode: str = "cos"

    # Loss weights
    feature_l1_weight: float = 30.0
    feature_l2_weight: float = 0.0
    mel_l1_weight: float = 30.0
    mel_l2_weight: float = 0.0
    adv_weight: float = 1.0
    feature_matching_weight: float = 10.0

    # GAN settings
    use_discriminator: bool = False
    adv_loss_type: Literal["hinge", "least_square"] = "hinge"  # Type of adversarial loss
    discriminator_lr: float | None = None  # Learning rate for discriminator
    discriminator_start_step: int = 0  # Step to start training discriminator
    discriminator_update_prob: float = 1.0  # Probability of updating discriminator at each step

    # Checkpoint loading
    ckpt_path: str | None = None  # Path to checkpoint to load from
    skip_loading_modules: tuple[str, ...] = ()  # Modules to skip when loading checkpoint

    # Other settings
    log_mel_samples: int = 10
    use_torch_compile: bool = True


class KanadePipeline(L.LightningModule):
    """LightningModule wrapper for KanadeModel, handling training (including GAN)."""

    def __init__(
        self,
        model_config: KanadeModelConfig,
        pipeline_config: KanadePipelineConfig,
        ssl_feature_extractor: SSLFeatureExtractor,
        local_encoder: Transformer,
        local_quantizer: FiniteScalarQuantizer,
        feature_decoder: Transformer | None,
        global_encoder: GlobalEncoder,
        mel_prenet: Transformer,
        mel_decoder: Transformer,
        mel_postnet: PostNet,
        discriminator: SpectrogramDiscriminator | None = None,
    ):
        super().__init__()
        self.config = pipeline_config
        self.save_hyperparameters("model_config", "pipeline_config")
        self.strict_loading = False
        self.automatic_optimization = False
        self.torch_compiled = False

        # Validate components required for training
        assert not pipeline_config.train_feature or feature_decoder is not None, (
            "Feature decoder must be provided if training feature reconstruction"
        )
        logger.info(
            f"Training configuration: train_feature={pipeline_config.train_feature}, train_mel={pipeline_config.train_mel}"
        )

        # 1. Kanade model
        self.model = KanadeModel(
            config=model_config,
            ssl_feature_extractor=ssl_feature_extractor,
            local_encoder=local_encoder,
            local_quantizer=local_quantizer,
            feature_decoder=feature_decoder,
            global_encoder=global_encoder,
            mel_decoder=mel_decoder,
            mel_prenet=mel_prenet,
            mel_postnet=mel_postnet,
        )
        self._freeze_unused_modules(pipeline_config.train_feature, pipeline_config.train_mel)

        # Calculate padding for expected SSL output length
        self.padding = self.model._calculate_waveform_padding(pipeline_config.audio_length)
        logger.info(f"Input waveform padding for SSL feature extractor: {self.padding} samples")

        # Calculate target mel spectrogram length
        self.target_mel_length = self.model._calculate_target_mel_length(pipeline_config.audio_length)
        logger.info(f"Target mel spectrogram length: {self.target_mel_length} frames")

        # 2. Discriminator
        self._init_discriminator(pipeline_config, discriminator)

        # 3. Mel spectrogram feature extractor for loss computation
        if pipeline_config.train_mel:
            self.mel_spec = MelSpectrogramFeature(
                sample_rate=model_config.sample_rate,
                n_fft=model_config.n_fft,
                hop_length=model_config.hop_length,
                n_mels=model_config.n_mels,
                padding=model_config.padding,
            )

        # Mel sample storage for logging
        self.vocoder = None
        self.validation_examples = []
        self.log_mel_samples = pipeline_config.log_mel_samples

    def _freeze_unused_modules(self, train_feature: bool, train_mel: bool):
        model = self.model
        if not train_feature:
            # Freeze local branch components if not training feature reconstruction
            freeze_modules([model.local_encoder, model.local_quantizer, model.feature_decoder])
            if model.conv_downsample is not None:
                freeze_modules([model.conv_downsample, model.conv_upsample])
            logger.info("Feature reconstruction branch frozen: local_encoder, local_quantizer, feature_decoder")

        if not train_mel:
            # Freeze global branch and mel generation components if not training mel generation
            freeze_modules(
                [model.global_encoder, model.mel_prenet, model.mel_conv_upsample, model.mel_decoder, model.mel_postnet]
            )
            logger.info(
                "Mel generation branch frozen: global_encoder, mel_prenet, mel_conv_upsample, mel_decoder, mel_postnet"
            )
            
    def setup(self, stage: str):
        # Torch compile model if enabled
        if torch.__version__ >= "2.0" and self.config.use_torch_compile:
            self.model = torch.compile(self.model)
            if self.discriminator is not None:
                self.discriminator = torch.compile(self.discriminator)
            self.torch_compiled = True

        # Load checkpoint if provided
        if self.config.ckpt_path:
            ckpt_path = self.config.ckpt_path

            # Download weights from HuggingFace Hub if needed
            if ckpt_path.startswith("hf:"):
                from huggingface_hub import hf_hub_download

                repo_id = ckpt_path[len("hf:") :]
                # Separate out revision if specified
                revision = None
                if "@" in repo_id:
                    repo_id, revision = repo_id.split("@", 1)

                ckpt_path = hf_hub_download(repo_id, filename="model.safetensors", revision=revision)

            self._load_weights(ckpt_path)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
            ssl_real: Extracted SSL features for local branch (B, T, C)
            ssl_recon: Reconstructed SSL features (B, T, C) - only if train_feature=True
            mel_recon: Generated mel spectrogram (B, n_mels, T) - only if train_mel=True
            loss_dict: Dictionary with auxiliary information (codes, losses, etc.)
        """
        loss_dict = {}

        # 1. Extract SSL features
        local_ssl_features, global_ssl_features = self.model.forward_ssl_features(waveform, padding=self.padding)

        # 2. Content branch processing
        content_embeddings, _, ssl_recon, perplexity = self.model.forward_content(local_ssl_features)
        loss_dict["local/perplexity"] = perplexity

        # 3. Global branch processing and mel reconstruction
        mel_recon = None
        if self.config.train_mel:
            global_embeddings = self.model.forward_global(global_ssl_features)
            mel_recon = self.model.forward_mel(content_embeddings, global_embeddings, mel_length=self.target_mel_length)

        return local_ssl_features, ssl_recon, mel_recon, loss_dict

    def _setup_vocoder(self):
        try:
            from vocos import Vocos

            model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
            return model.eval()
        except ImportError:
            logger.error("Vocos not found. Please install vocos to enable vocoding during validation/prediction.")
            return None

    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        self.vocoder = self.vocoder.to(mel.device)
        mel = mel.float()
        waveform = self.vocoder.decode(mel)  # (B, T)

        return waveform.cpu().float()

    def on_validation_start(self):
        self.vocoder = self._setup_vocoder()

    def on_predict_start(self):
        self.vocoder = self._setup_vocoder()

    def on_validation_end(self):
        if len(self.validation_examples) > 0:
            for i, (mel_real, mel_recon, audio_real, audio_gen) in enumerate(self.validation_examples):
                # Log spectrograms
                fig_real = self._get_spectrogram_plot(mel_real)
                fig_gen = self._get_spectrogram_plot(mel_recon)
                self._log_figure(f"val/{i}_mel_real", fig_real)
                self._log_figure(f"val/{i}_mel_gen", fig_gen)

                # Log audio samples
                if audio_gen is not None:
                    audio_real = audio_real.cpu().numpy()
                    audio_gen = audio_gen.cpu().numpy()
                    self._log_audio(f"val/{i}_audio_real", audio_real)
                    self._log_audio(f"val/{i}_audio_gen", audio_gen)

            self.validation_examples = []

        # Clear vocoder to free memory
        self.vocoder = None

    def _log_figure(self, tag: str, fig):
        """Log a matplotlib figure to the logger."""
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_figure(tag, fig, self.global_step)
        elif isinstance(self.logger, WandbLogger):
            import PIL.Image as Image

            fig.canvas.draw()
            image = Image.frombytes("RGBa", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
            image = image.convert("RGB")
            self.logger.log_image(tag, [image], step=self.global_step)

    def _log_audio(self, tag: str, audio: np.ndarray):
        """Log an audio sample to the logger."""
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_audio(tag, audio, self.global_step, sample_rate=self.model.config.sample_rate)
        elif isinstance(self.logger, WandbLogger):
            self.logger.log_audio(
                tag, [audio.flatten()], sample_rate=[self.model.config.sample_rate], step=self.global_step
            )

    def _get_spectrogram_plot(self, mel: torch.Tensor):
        from matplotlib import pyplot as plt

        mel = mel.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma", vmin=-8.0, vmax=5.0)
        fig.colorbar(im, ax=ax)
        ax.set_ylabel("Mel bins")
        ax.set_xlabel("Time steps")
        fig.tight_layout()
        return fig

    def _load_weights(self, ckpt_path: str | None, model_state_dict: dict[str, torch.Tensor] | None = None):
        """Load model and discriminator weights from checkpoint. Supports .ckpt (Lightning), .safetensors, .pt/.pth formats.
        If model_state_dict is provided, load weights from it instead of ckpt_path."""

        def select_keys(state_dict: dict, prefix: str) -> dict:
            """Select keys from state_dict that start with the given prefix. Remove the prefix from keys."""
            return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

        def remove_prefix(state_dict: dict, prefix: str) -> dict:
            """Remove a prefix from keys that start with that prefix."""
            return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

        def add_prefix(state_dict: dict, prefix: str) -> dict:
            """Add a prefix to keys that do not start with that prefix."""
            return {f"{prefix}{k}" if not k.startswith(prefix) else k: v for k, v in state_dict.items()}

        # Load state dict
        if model_state_dict is not None:
            # Load from provided state dict
            disc_state_dict = {}
        elif ckpt_path.endswith(".ckpt"):
            # Lightning checkpoint
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model_state_dict = select_keys(checkpoint["state_dict"], "model.")
            disc_state_dict = select_keys(checkpoint["state_dict"], "discriminator.")
        elif ckpt_path.endswith(".safetensors"):
            # Safetensors checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(ckpt_path, device="cpu")
            model_state_dict = checkpoint
            disc_state_dict = {}
        elif ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
            # Standard PyTorch checkpoint
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model_state_dict = checkpoint
            disc_state_dict = {}
        else:
            raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

        # Load model weights
        model_state_dict = remove_prefix(model_state_dict, "_orig_mod.")
        model_state_dict = {
            k: v
            for k, v in model_state_dict.items()
            if not any(k.startswith(module) for module in self.config.skip_loading_modules)
        }
        if self.torch_compiled:
            model_state_dict = add_prefix(model_state_dict, "_orig_mod.")

        if len(model_state_dict) > 0:
            result = self.model.load_state_dict(model_state_dict, strict=False)
            logger.info(f"Loaded model weights from {ckpt_path or 'provided state_dict'}.")
            if result.missing_keys:
                logger.debug(f"Missing keys in model state_dict: {result.missing_keys}")
            if result.unexpected_keys:
                logger.debug(f"Unexpected keys in model state_dict: {result.unexpected_keys}")

        # Load discriminator weights if available
        if self.use_discriminator:
            disc_state_dict = remove_prefix(disc_state_dict, "_orig_mod.")
            if self.torch_compiled:
                disc_state_dict = add_prefix(disc_state_dict, "_orig_mod.")

            if len(disc_state_dict) > 0:
                result = self.discriminator.load_state_dict(disc_state_dict, strict=False)
                logger.info(f"Loaded discriminator weights from {ckpt_path}.")
                if result.missing_keys:
                    logger.debug(f"Missing keys in discriminator state_dict: {result.missing_keys}")
                if result.unexpected_keys:
                    logger.debug(f"Unexpected keys in discriminator state_dict: {result.unexpected_keys}")

    @classmethod
    def from_hparams(cls, config_path: str) -> "KanadePipeline":
        """Instantiate KanadePipeline from config file.
        Args:
            config_path (str): Path to model configuration file (.yaml).
        Returns:
            KanadePipeline: Instantiated KanadePipeline.
        """
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Remove related fields to prevent loading actual weights here
        new_config = {"model": config["model"]}
        pipeline_config = new_config["model"]["init_args"]["pipeline_config"]
        if "ckpt_path" in pipeline_config:
            del pipeline_config["ckpt_path"]
        if "skip_loading_modules" in pipeline_config:
            del pipeline_config["skip_loading_modules"]

        # Instantiate model using jsonargparse
        parser = jsonargparse.ArgumentParser(exit_on_error=False)
        parser.add_argument("--model", type=KanadePipeline)
        cfg = parser.parse_object(new_config)
        cfg = parser.instantiate_classes(cfg)
        return cfg.model

    @staticmethod
    def from_pretrained(config_path: str, ckpt_path: str) -> "KanadePipeline":
        """Load KanadePipeline from training configuration and checkpoint files.
        Args:
            config_path: Path to pipeline configuration file (YAML).
            ckpt_path: Path to checkpoint file (.ckpt) or model weights (.safetensors).
        Returns:
            KanadePipeline: Instantied KanadePipeline with loaded weights.
        """
        # Load pipeline from config
        model = KanadePipeline.from_hparams(config_path)
        # Load the weights
        model._load_weights(ckpt_path)
        return model
