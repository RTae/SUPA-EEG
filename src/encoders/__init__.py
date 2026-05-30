"""EEG and visual encoder modules."""

from .eegnet_encoder import EEGNetEncoder
from .visual_encoder import VisualEncoder, VisualFeatureLookup

__all__ = ["EEGNetEncoder", "VisualEncoder", "VisualFeatureLookup"]
