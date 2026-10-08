"""tracer: attribute images to the image autoregressive (IAR) model that generated them, or flag them as outliers.

Modules:
    common   paths, label mapping, data index, image loading, and loaders for the RAR / VAR autoencoders
    signals  provenance signals from Zhao et al. (QuantLoss, EncLoss, calibrated EncLoss, ...) for one batch
    stage1   NearestFamilyThreshold, the Stage 1 decision rule (family -> outlier -> size)
    metrics  accuracy plus the family / outlier / size breakdown, and confusion matrices

See tracer/README.md for usage.
"""
