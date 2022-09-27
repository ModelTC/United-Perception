from .registry import Registry

# model
MODULE_ZOO_REGISTRY = Registry()
MODULE_PROCESS_REGISTRY = Registry()
MODULE_WRAPPER_REGISTRY = Registry()
MODEL_WRAPPER_REGISTRY = Registry()
EMA_REGISTRY = Registry()

# data
DATASET_REGISTRY = Registry()
DATALOADER_REGISTRY = Registry()
BATCH_SAMPLER_REGISTRY = Registry()
AUGMENTATION_REGISTRY = Registry()
BATCHING_REGISTRY = Registry()

# predictor
ROI_PREDICTOR_REGISTRY = Registry()
BBOX_PREDICTOR_REGISTRY = Registry()
MASK_PREDICTOR_REGISTRY = Registry()

# supervisior
ROI_SUPERVISOR_REGISTRY = Registry()
BBOX_SUPERVISOR_REGISTRY = Registry()
MASK_SUPERVISOR_REGISTRY = Registry()

# matcher
MATCHER_REGISTRY = Registry()

# sampler
ROI_SAMPLER_REGISTRY = Registry()
SAMPLER_REGISTRY = Registry()

# merger
ROI_MERGER_REGISTRY = Registry()

# lr
WARM_LR_REGISTRY = Registry()
LR_REGISTRY = Registry()

# evaluator
EVALUATOR_REGISTRY = Registry()

# loss
LOSSES_REGISTRY = Registry()

# image reader
IMAGE_READER_REGISTRY = Registry()

# hook
HOOK_REGISTRY = Registry()

# saver
SAVER_REGISTRY = Registry()

# anchor generate
ANCHOR_GENERATOR_REGISTRY = Registry()

# mask target generate
MASK_GENERATOR_REGISTRY = Registry()

# subcommand
SUBCOMMAND_REGISTRY = Registry()

# initializer
INITIALIZER_REGISTRY = Registry()

# runner
RUNNER_REGISTRY = Registry()

# inferencer
INFERENCER_REGISTRY = Registry()
VISUALIZER_REGISTRY = Registry()

# optimizer
OPTIMIZER_REGISTRY = Registry()

LR_SCHEDULER_REGISTY = Registry()

WARM_SCHEDULER_REGISTY = Registry()

DATA_BUILDER_REGISTY = Registry()

MODEL_HELPER_REGISTRY = Registry()

# deploy
DEPLOY_REGISTRY = Registry()
TOONNX_REGISTRY = Registry()

# distill
MIMIC_REGISTRY = Registry()
MIMIC_LOSS_REGISTRY = Registry()

# box_coder
BOX_CODER_REGISTRY = Registry()
