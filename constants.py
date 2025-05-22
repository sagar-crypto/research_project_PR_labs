# constants.py

# ── BatchNorm defaults ─────────────────────────────────────────────────────────
BN_EPS      = 0.001
BN_MOMENTUM = 0.99

# ── Dropout defaults ───────────────────────────────────────────────────────────
USE_DROPOUT   = True
DROPOUT_RATE  = 0.2

# ── Encoder conv‐block specs in order ─────────────────────────────────────────
# each dict is (out_channels, kernel_size, padding, stride)
ENCODER_BLOCKS = [
    {"out_ch":  64, "kernel": 21, "pad": 10, "stride": 1},
    {"out_ch":  64, "kernel": 15, "pad":  7, "stride": 2},
    {"out_ch": 128, "kernel": 11, "pad":  5, "stride": 2},
    {"out_ch": 128, "kernel":  7, "pad":  3, "stride": 2},
]

# how many time‐steps remaining after these 4 blocks, adapt to 4
ADAPT_POOL_OUTPUT_SIZE = 4

# ── Decoder “deconv” blocks ───────────────────────────────────────────────────
# we mirror the encoder, but with is_conv=False
DECODER_BLOCKS = [
    {"out_ch":  64, "kernel":  7, "pad":  3},
    {"out_ch":  32, "kernel": 11, "pad":  5},
    {"out_ch":  32, "kernel": 15, "pad":  7},
    {"out_ch":  16, "kernel": 21, "pad": 10},
]

# ── Final conv in decoder ─────────────────────────────────────────────────────
FINAL_OUT_CHANNELS = None  # fill in at runtime (should equal in_channels)
FINAL_KERNEL        = 21
FINAL_PADDING       = 10
