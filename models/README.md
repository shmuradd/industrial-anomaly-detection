# Fitted normal model

`bottle_gaussian.npz` contains float32 means and regularized variances (192 × 32 × 32), plus the calibration threshold. It was fitted to 167 normal MVTec AD bottle training images, with a separate 42-normal calibration set. No test image was fitted.

The model uses the frozen TorchVision ResNet18 IMAGENET1K_V1 encoder. The encoder weights download once at inference. Run `inspect-bottles predict --image <image.png>` from the repository root. Scores are anomaly distances, not probabilities.

SHA-256: `ac49c69462fe58c72db7fe5add1c9d2b0a47129c98757b0e8744dcdee5467aa3`.

See `NOTICE.md` for data-derived artifact terms and attribution.
