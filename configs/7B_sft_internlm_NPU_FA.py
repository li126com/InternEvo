JOB_NAME = "7b_train"
DO_ALERT = False

SEQ_LEN = 1024
HIDDEN_SIZE = 2048
NUM_ATTENTION_HEAD = 16
MLP_RATIO = 8 / 3
NUM_LAYER = 20
VOCAB_SIZE = 103168

MODEL_ONLY_FOLDER = "local:llm_ckpts_ljx/xxxx"
SAVE_CKPT_FOLDER = "local:llm_ckpts_wgt_1b"
LOAD_CKPT_FOLDER = "local:llm_ckpts_ljx/49"

TRAIN_FOLDER = "/data/wangguoteng/opensource_data/0711_scratch_tokenized_refineweb_small/train/"  # "/path/to/dataset"
VALID_FOLDER = None  # "/path/to/dataset"

use_flash_attn_npu = True
pack_sample_into_one = False
CHECKPOINT_EVERY = 500
ckpt = dict(
    enable_save_ckpt=False,  # enable ckpt save.
    save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
    auto_resume=False,
    load_ckpt_info=dict(path=None, content=("model",), ckpt_type="internlm"),
    checkpoint_every=CHECKPOINT_EVERY,
    async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),  # snapshot ckpt save frequency.
)

data = dict(
    seq_len=SEQ_LEN,
    micro_num=2,
    micro_bsz=2,
    valid_micro_num=4,
    valid_every=1000000,
    pack_sample_into_one=pack_sample_into_one,
    total_steps=2000,
    skip_batches="",
    rampup_batch_size="",
    min_length=50,
    train_folder=TRAIN_FOLDER,
    valid_folder=VALID_FOLDER,
    empty_cache_and_diag_interval=200,
    diag_outlier_ratio=1.1,
    pack_type="TorchAttnMask",     # TorchAttnMask, TorchUnPacked
)
grad_scaler = dict(
    fp16=dict(
        initial_scale=2**16,
        min_scale=1,
        growth_interval=1000,
    ),
    growth_factor=2,
    backoff_factor=0.5,
    max_scale=2**24,
    hysteresis=2,
)
hybrid_zero_optimizer = dict(
    overlap_sync_grad=True,
    overlap_sync_param=False,
    reduce_bucket_size=512 * 1024 * 1024,
    clip_grad_norm=1.0,
)

loss = dict(
    label_smoothing=0,
)
adam = dict(
    lr=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
)
lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=0,  # optimizer_warmup_step
    warmup_ratio=0.01,
    eta_min=1e-5,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)
use_fp32_norm = False
model = dict(
    checkpoint=False,  # The proportion of layers for activation aheckpointing, the optional value are True/False/[0-1]
    num_attention_heads=NUM_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    dtype="torch.bfloat16",  # Support: "torch.float16", "torch.half", "torch.bfloat16", "torch.float32", "torch.tf32"
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-5,
    use_flash_attn=False,
    use_flash_attn_npu=use_flash_attn_npu,
    num_chunks=1,  # if num_chunks > 1, interleaved pipeline scheduler is used.
)
parallel = dict(
    zero1=dict(size=-1),
    tensor=dict(size=1, mode="mtp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=1, overlap=True, memory_pool=True),
)
cudnn_deterministic = False
cudnn_benchmark = False
monitor = dict(
    alert=dict(
        enable_feishu_alert=DO_ALERT,
        feishu_alert_address=None,  # feishu webhook to send alert message
        light_monitor_address=None,  # light_monitor address to send heartbeat
        alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
    ),
    tensorboard=dict(
        queue_max_length=10,
    ),
)