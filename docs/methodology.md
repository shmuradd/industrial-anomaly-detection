# Methodology

For a channel c and grid position p, estimate μ[c,p] and unbiased sample variance v[c,p] from the fitting normals. Regularize with 0.01 × max(mean spatial variance of channel c, 10⁻⁶). The feature distance is sqrt(mean_c((x[c,p] − μ[c,p])² / v_regularized[c,p])). This is a channel-normalized diagonal Mahalanobis distance.

Bilinearly upsample the 32 × 32 distance map to 256 × 256, then apply a Gaussian filter with sigma 4 pixels. The image score averages the largest ceil(0.01 × 256 × 256) distances. Maps have a common numeric scale; per-image rescaling is used neither for scoring nor evaluation.

Training order is sorted filenames followed by NumPy default_rng(42).permutation. The first ceil(0.2 × 209) images calibrate; the remaining 167 fit. Threshold rank is ceil((n+1)(1−alpha)) with n=42 and alpha=0.05: rank 41 in ascending scores. A score equal to the threshold is accepted as normal. This split-conformal-style bound assumes exchangeable future normal images; camera or illumination shifts can violate that assumption. No guarantee is claimed for the specific 20 test normals.

The neural encoder is frozen and runs in evaluation mode. No gradients, optimizer, augmentation, or neural training epochs are involved. The fitted model stores spatial means, variances, and threshold, while the official TorchVision backbone is reused at inference.

The test set contains 20 good, 20 broken_large, 22 broken_small, and 21 contamination images. Image AUROC and average precision use continuous scores. Threshold metrics use the previously fixed normal-calibration threshold. Pixel AUROC pools every test pixel, including normal-image backgrounds. Masks are resized with nearest-neighbor interpolation, images with bilinear interpolation. Pixel evaluation is at 256 × 256 and does not report AU-PRO, Dice, or an optimized segmentation threshold.

The report stores all 83 image predictions, image hashes, all 209 split assignments, calibration scores, and archive provenance. Hash sets for fitting, calibration, and test images have no overlap. Configuration was fixed before the first test run; this repository reports that run without test-driven tuning. No patient, medical, or personally identifiable data is involved.

Reproducibility depends on the recorded package versions, weights, CPU floating-point behavior, and archive. Small floating-point differences can occur on other hardware. The notebook links the committed figures rather than embedding large copies of them.
