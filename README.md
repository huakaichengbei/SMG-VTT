# SMG-VTT

A modular, research-oriented implementation of the method described in *Semantic-Aware Visual–Tactile Graph Fusion with Class-Level Textual Prototype Matching* (supplied manuscript). 

## Scope

- **Visual**: ResNet18 feature map, ORB salient points, bilinear sampling, feature-space KNN with Gaussian weights, three residual weighted GCN layers, global max pooling.
- **TiG**: tactile patch nodes; per-layer rebuilt feature-space KNN; Max-Relative GraphConv; independent multi-head aggregation; FFN; global max pooling.
- **Text**: the exact manuscript category-name prompt template, offline description JSON, DistilBERT [CLS]-position representation, learnable projection to 256-D and normalized full prototype bank.
- **SGPM**: candidate-conditioned modality gate, normalized fused features, all-category inference.


## Explanation

The supplied paper specifies essential components but not every implementation choice. This release defaults to ORB, tactile 14×14 grid, linear integer-rounded k schedule, average-pooled tactile patches, Gaussian visual edge scaling, 224×224 image resizing in helper functions, and a particular residual placement/normalization in graph blocks.


## Inference contract

`SMGVTT.forward(visual_image, tactile_image, prototypes)` takes `[B,3,H,W]`, `[B,C_t,H,W]`, and **the entire** `[n_classes,256]` prototype bank. The module outputs one score per candidate. Labels are not used to pick prototypes. `predict` maps argmax to the order of names stored in `PrototypeBank`.


