"""RoboMME multi-camera spatial SigLIP feature encoding & cached dataset loaders."""

from .cache_dataset import (
    RoboMMEChunkDataset,
    RoboMMEEpisodeDataset,
    build_padded_action_chunk,
    load_cache_normalizer,
    load_task_shards,
)
from .constants import (
    DELTA_JOINT_MASK,
    LEROBOT_BLOCK_TO_TASK,
    ROBOMME_SUITES,
    ROBOMME_TASKS,
    TASKS_WITH_VIDEO_DEMO,
    TASK_TO_LEROBOT_BLOCK,
    TASK_TO_SUITE,
    epis_idx_to_task,
    task_to_epis_range,
)
from .vision_encoder import (
    SIGLIP_MODEL_ID,
    SiglipFeatureEncoder,
    pool_siglip_patch_tokens,
    preprocess_siglip_rgb,
)

__all__ = [
    "DELTA_JOINT_MASK",
    "LEROBOT_BLOCK_TO_TASK",
    "ROBOMME_SUITES",
    "ROBOMME_TASKS",
    "RoboMMEChunkDataset",
    "RoboMMEEpisodeDataset",
    "SIGLIP_MODEL_ID",
    "SiglipFeatureEncoder",
    "TASKS_WITH_VIDEO_DEMO",
    "TASK_TO_LEROBOT_BLOCK",
    "TASK_TO_SUITE",
    "build_padded_action_chunk",
    "epis_idx_to_task",
    "load_cache_normalizer",
    "load_task_shards",
    "pool_siglip_patch_tokens",
    "preprocess_siglip_rgb",
    "task_to_epis_range",
]
