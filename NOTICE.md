# Data and artifact attribution

MVTec AD bottle image data: © MVTec Software GmbH. Dataset authors: Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger, “MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” CVPR 2019.

Original source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International, https://creativecommons.org/licenses/by-nc-sa/4.0/

`reports/bottle/inspection.png` modifies MVTec images by resizing them and adding reference masks, score overlays, and labels. This image-containing figure is provided under CC BY-NC-SA 4.0. The fitted data-derived statistics in `models/bottle_gaussian.npz` are supplied for noncommercial research under the same terms. No endorsement by MVTec is implied.

The mirror is a download convenience, not the dataset creator. Its exact downloaded archive hash and URL are recorded in `reports/bottle/provenance.json`. Raw images and pretrained ResNet18 weights are not redistributed in the repository. TorchVision obtains its pretrained weights separately.
