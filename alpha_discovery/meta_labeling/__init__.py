# alpha_discovery/meta_labeling/__init__.py
"""
Meta-Labeling System for Options-Based Trading

This module implements a meta-labeling filter that sits on top of gauntlet-passing
setups to decide whether to take or skip individual trades based on entry-time
features and recent performance.
"""

from .core import MetaLabelingSystem
from .types import MetaLabelingResults
from .features import MetaFeatureExtractor
from .models import MetaModelTrainer
from .evaluation import MetaEvaluator
from .artifacts import MetaArtifactGenerator

__all__ = [
    'MetaLabelingSystem',
    'MetaLabelingResults', 
    'MetaFeatureExtractor',
    'MetaModelTrainer',
    'MetaEvaluator',
    'MetaArtifactGenerator'
]
