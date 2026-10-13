# Empatia_FLAGS

**Self-supervised physiological graph learning for label-efficient emotion recognition.**

Empatia_FLAGS contains research code for learning physiological representations from hierarchical feature trees and transferring those representations to an emotion classifier. It provides signal-specific encoders for blood volume pulse (BVP), galvanic skin response (GSR), and skin temperature (SKT), together with frozen-encoder classification, cluster-conditioned pretraining, and few-shot user-adaptation experiments.

This repository corresponds to **Chapter 5 — Overcoming Label Scarcity via Self-Supervised Graph Learning** of Junjiao Sun's doctoral thesis, *Toward Practical and Uncertainty-Aware Affective Computing with Wearable Physiological Signals*. Related work is presented in:

> Junjiao Sun, J. Miranda, Jorge Portilla, and Andrés Otero. **From Cloud-Heavy to Edge-Ready: Self-supervised Transfer-efficient Emotion Recognition.** *2026 Design, Automation & Test in Europe Conference (DATE)*, pp. 1–3, 2026. [DOI: 10.23919/DATE69613.2026.11539461](https://doi.org/10.23919/DATE69613.2026.11539461).

The companion repositories [Empatia_DL](https://github.com/des-cei/Empatia_DL), [Empatia_CNN-LSTM](https://github.com/des-cei/Empatia_CNN-LSTM), and [Empatia_Cluster](https://github.com/des-cei/Empatia_Cluster) cover feature-map recognition and cluster-based personalization developed in Chapters 3–4.

> **Implementation status:** The repository contains development-time experiment scripts, primarily configured for WEMAC binary fear recognition. Feature caches, normalization logs, and cluster-assignment CSV files must be supplied separately. Some defaults select ablation models, and several training details require correction before end-to-end experiments. See **Required adjustments** below before running the commands.

## Method overview

Emotion annotation is expensive, particularly when adapting a wearable system to a new user. This framework separates representation learning from supervised task adaptation:

1. Extract the same 123 physiological descriptors used in the earlier chapters.
2. Organize each time window into three signal-specific hierarchical trees.
3. Pretrain an encoder for each signal using structural pseudo-labels derived from its tree.
4. Freeze the pretrained encoders and combine their node representations.
5. Train a compact emotion classifier using labeled data, optionally adapting it to a new user with a small labeled subset.

The pretext task predicts a leaf node's **physiological parent group**, rather than its emotion label. Emotion annotations are used in the downstream classification stage. The supplied cache loader still expects annotated WEMAC records, so using a completely unannotated collection requires adapting the input interface even though the pretext loss itself does not use emotion labels.

Chapter 5 also describes cloud-side user clustering, centroid-based encoder selection, and edge-side adaptation. The repository consumes external cluster assignments and saves/loads local encoder files; it does not include a complete centroid-generation, online assignment, or cloud-download service. Although the thesis evaluates WEMAC and WESAD, the released data interfaces are WEMAC-specific.

## Physiological trees and self-supervision

`Train_GCN/Generate_tree.py` builds a PyTorch Geometric `Data` object for each signal in each feature frame.

| Signal | Descriptor slice | Leaf nodes | Parent groups | Total nodes | Encoder output per graph |
| --- | --- | --- | --- | --- | --- |
| BVP | `0:84` | 84 | 7 | 92 | `92 × 7` |
| GSR | `84:118` | 34 | 6 | 41 | `41 × 6` |
| SKT | `118:123` | 5 | 2 | 8 | `8 × 2` |

Each tree contains one root, intermediate group nodes, and descriptor leaves. BVP groups are PP, HR, PW, PDT, PRT, PA, and PWR. GSR groups cover signal statistics, PSD, peaks, first differences, SCL, and SCR. SKT uses two descriptor groups.

Each node has a scalar input feature. Leaf values come from the normalized descriptors. The checked-in implementation initializes root/group nodes with integer identifiers and supplies parent-to-child edges only. **This differs from the thesis formulation**, which initializes internal nodes by averaging their children. Explicit reverse edges would also be needed if bidirectional parent–child message passing is intended.

The `y` field contains leaf-level parent-group targets: seven classes for BVP, six for GSR, and two for SKT. `train_mask` currently stores integer leaf indices. These structural labels must remain separate from the binary emotion labels returned by `Get_emotion_label`.

### Encoders

`Train_GCN/Model_GCN.py` defines:

- **GCN:** three `GCNConv` layers, with ReLU after the first two layers and an unactivated final output for the classification loss. The standard configuration is `GCN(1, 32, C)`, where `C` is the number of parent groups.
- **GAT:** an alternative two-layer graph-attention encoder.

`Train_GCN/Replacement_ablation/` contains node-wise MLP, 1D-CNN, and Transformer alternatives and their experiment drivers. These alternatives allow comparison with representations learned without graph convolution.

**Current default:** `train_BVP` instantiates `NodeTransformer`, while `train_GSR` and `train_SKT` instantiate GCNs. Select `GCN(1, 32, 7)` in `Train_BVP_GCN.py` for a three-GCN configuration. Script and checkpoint names containing `GCN` do not guarantee that every active encoder is a GCN.

## Frozen-encoder emotion classification

Both combined models freeze the three encoder parameter sets with `requires_grad=False`.

| Stage | Tensor shape / operation |
| --- | --- |
| BVP compression | `B × 92 × 7` → Conv1D with kernel 7 → `B × 92 × 1`. |
| GSR compression | `B × 41 × 6` → Conv1D with kernel 6 → `B × 41 × 1`. |
| SKT compression | `B × 8 × 2` → Conv1D with kernel 2 → `B × 8 × 1`. |
| Normalization and fusion | Batch normalization, concatenation, and squeeze → `B × 141`. |
| `CombinedGCN_cnn` | Dense binary classifier. |
| `CombinedGCN_cnn_lstm` | LSTM followed by a dense classifier. |

Here, graph nodes are the convolution channels and encoder dimensions are the convolution's length axis. The convolutions compress each signal's embedding width before fusion.

The LSTM implementation currently receives a two-dimensional `B × 141` tensor. PyTorch treats this as one unbatched sequence, so samples within the loader batch become sequence steps. This is different from the thesis table's `B × 1 × 141` layout. Define the intended sequence layout explicitly before using this variant; chronological multi-window modeling requires constructing actual ordered sequences.

## Repository guide

| Location | Purpose |
| --- | --- |
| `Feature_extraction/` | BVP, GSR, and SKT descriptor functions. |
| `Feature_wise_Normalization/` | Feature-cache loading, outlier processing, normalization, and legacy map utilities. |
| `FWN_main/` | Bundled feature-wise normalization routines. |
| `Abnormal_analyze/` | Statistical and outlier-analysis utilities. |
| `Train_GCN/Generate_tree.py` | Build signal trees and retrieve emotion labels. |
| `Train_GCN/Model_GCN.py` | GCN and GAT definitions. |
| `Train_GCN/Train_BVP_GCN.py`, `Train_GSR_GCN.py`, `Train_SKT_GCN.py` | Signal-specific pretext training functions. |
| `Train_GCN/CombinedGCN_cnn.py`, `CombinedGCN_cnn_lstm.py` | Frozen-encoder downstream classifiers. |
| `Train_GCN/Train_Frozen_Classification.py` | Combined pretraining/classification experiment without external cluster assignments. |
| `Train_GCN/Train_Frozen_Classification_cluster.py` | CSV-driven cluster-specific subject-holdout experiment. |
| `Train_GCN/Train_Cloud.py`, `Train_Cloud_cluster.py` | Save encoder sets for held-out-user experiments. |
| `Train_GCN/Train_Edge.py`, `Train_Edge_cluster.py` | Load encoder sets and adapt a classifier on user data. |
| `Train_GCN/Replacement_ablation/` | MLP, 1D-CNN, and Transformer comparisons. |

## Installation and input preparation

```bash
git clone https://github.com/des-cei/Empatia_FLAGS.git
cd Empatia_FLAGS
python -m venv .venv
source .venv/bin/activate
export EMPATIA_FLAGS_ROOT="$PWD"
export PYTHONPATH="$EMPATIA_FLAGS_ROOT:$EMPATIA_FLAGS_ROOT/Train_GCN${PYTHONPATH:+:$PYTHONPATH}"
```

Install compatible PyTorch, torchvision, and PyTorch Geometric packages for your Python/CUDA environment. Additional dependencies inferred from imports include:

```bash
python -m pip install numpy scipy pandas scikit-learn matplotlib \
    mat73 biosppy distfit thop
```

`thop` is imported by the ablation modules and the cluster-classification profiler. A pinned environment is not provided. The thesis reports an NVIDIA A30 training environment and Jetson Orin Nano deployment with JetPack 6.1, but those descriptions do not establish a tested dependency lockfile for this checkout.

Training functions select CUDA when available and otherwise CPU. Existing serialized models may need explicit device mapping when moved between machines. Replace hard-coded development paths such as `/home/junjiao/PycharmProjects/Empatia_GCN/` and run the experiment drivers from `Train_GCN/` so their default relative paths resolve.

### Feature cache

Supply `Feature_wise_Normalization/Pack_all_data.json`, containing participant → video → window dictionaries. Participants use identifiers such as `V001`. Window records contain descriptor values and `ReportadaBinarizado` emotion labels. The loader selects descriptor names containing `bvp`, `gsr`, or `skt`.

The graph builders require a fixed feature order:

| Rows in the normalized array | Content |
| --- | --- |
| `0:84` | BVP descriptors in the hierarchy's expected order. |
| `84:118` | GSR descriptors in the expected order. |
| `118:123` | SKT descriptors in the expected order. |
| `123` | Emotion label. |
| `124` | Numeric participant ID. |
| `125` | Numeric video ID. |

The resulting array is `126 × N`, with one column per frame. Descriptor order is inherited from the cache-loading process, so matching names and counts alone is insufficient: verify every column's ordering against `Generate_tree.py`. Do not apply arbitrary feature selection without rebuilding the topology and downstream dimensions.

The repository does not contain a complete raw-recording-to-cache exporter. Prepare a compatible cache externally, using the companion projects where appropriate, and verify recording-version and label compatibility.

Supply the corresponding `Feature_wise_Normalization/log_data_normalization_IT06_knn.log`. The parser expects a completed `Over all Best N:` entry listing transformations in feature order. Raw datasets and these generated preprocessing files are not bundled.

### Cluster assignments

Cluster-aware drivers expect `clustering_WEMAC_123feat_v03feb.csv` at the repository root, with comma-separated columns:

| Column | Meaning |
| --- | --- |
| `id_vol` | Integer participant ID. |
| `id_cluster` | Cluster assignment. |
| `method` | Role such as `method2_train` or `method2_test`. |
| `vol_test` | Held-out participant identifying the fold. |

For each `vol_test`, the helper takes the cluster ID from the last matching row and selects `method2_train` participants in that cluster. Ensure the final matching row represents the held-out user's assigned cluster, and exclude that user from the training records. Generate assignments using training subjects only and retain their provenance.

## Required adjustments

The following issues affect execution or learning and should be addressed before treating a run as an experiment:

| Issue | Required action |
| --- | --- |
| Overwritten split helper | `Data_extraction.py` defines `get_train_test` twice. The cluster-list version replaces the one-argument version still called by `data_extraction`. Give the helpers distinct names and update their callers; otherwise the non-cluster path fails with a missing-argument error. |
| Leaf indexing during batching | Integer indices stored as `train_mask` are not automatically offset per graph under standard PyG batching. Use a full-length Boolean leaf mask or implement explicit index-offset batching, keeping predictions aligned with leaf targets. See the [PyG batching guide](https://pytorch-geometric.readthedocs.io/en/2.6.0/advanced/batching.html). |
| Multimodal alignment | Downstream scripts zip four independently shuffled loaders for BVP, GSR, SKT, and labels. Use one paired dataset/shared ordering so all three graphs and the emotion label refer to the same frame. |
| Optimizer parameter selection | Drivers optimize `combined_model.fc.parameters()`, but `forward` uses `dense_layer_fc`, not `fc`. Include the active unfrozen classifier parameters in the optimizer; as written, the optimizer does not update the prediction path. |
| Loss and output contract | Both combined models end the active classifier with sigmoid while training uses `CrossEntropyLoss`. Return raw class logits for that loss, or consistently redesign the output/loss pair. |
| LSTM dimensions | Supply an explicit batch/sequence/feature layout and select the desired sequence output. Do not use randomly ordered batch samples as temporal steps. |
| Fixed batch size | Combined models reshape using a constructor-supplied batch size. Drivers use `drop_last=True`, including for testing, to match it. Derive batch size dynamically and retain all evaluation samples; small datasets can otherwise produce no batches. |
| Legacy standalone entry points | `Train_SKT_GCN.py` passes path strings to a function expecting arrays. Use the integrated drivers or load normalized arrays explicitly before calling `train_SKT`. |
| Frozen encoder mode | Parameter freezing does not disable dropout. When using Transformer ablations, keep frozen encoders in evaluation mode during classifier training if deterministic frozen features are intended. |

For example, after freezing the encoders, the active downstream parameters can be selected with:

```python
optimizer = torch.optim.Adam(
    (p for p in combined_model.parameters() if p.requires_grad),
    lr=1e-3,
)
```

This addresses optimizer selection only; the batching, alignment, output, and shape issues above still need to be resolved.

## Experiment workflows

These commands describe the intended entry points **after** input preparation and the required adjustments. Configure model choices and training settings in the scripts; there is no unified experiment CLI.

### Integrated pretraining and classification

```bash
cd "$EMPATIA_FLAGS_ROOT/Train_GCN"
python Train_Frozen_Classification_cluster.py
```

This driver reads cluster assignments, pretrains three signal encoders for each held-out-user case, freezes them, and trains the downstream classifier. The active classifier is `CombinedGCN_cnn_lstm`.

For the non-cluster experiment, use `Train_Frozen_Classification.py`, whose active classifier is `CombinedGCN_cnn`. Its initial split is a random participant-level split, not a complete LOSO loop.

### Cloud pretraining followed by user adaptation

```bash
python Train_Cloud_cluster.py
python Train_Edge_cluster.py
```

For the non-cluster alternative, use `Train_Cloud.py` followed by `Train_Edge.py`.

Cloud drivers exclude the target user and perform an additional approximately 80/20 participant split among the remaining users before encoder training. They save three encoders for each held-out user under:

```text
GCN_models/GCN_model_Vol_<ID>/gcn_bvp.pth
GCN_models/GCN_model_Vol_<ID>/gcn_gsr.pth
GCN_models/GCN_model_Vol_<ID>/gcn_skt.pth
```

Both cloud variants use the same output names. Keep separate output directories for clustered/non-clustered runs and encoder ablations to avoid overwriting or mixing experiments.

The edge drivers load those local files and use the **first 20% of the target user's frame columns** for supervised adaptation, with the remaining 80% for evaluation. This is a fraction-based split in cache order, not a balanced fixed-shot sampler. `Train_Edge_cluster.py` does not itself read the cluster CSV; its cluster-specific behavior depends on loading the encoders produced by the matching cloud run.

### Default training settings

| Setting | Pretext training | Downstream classification |
| --- | --- | --- |
| Optimizer | Adam | Adam, after correcting parameter selection. |
| Learning rate | `0.01` | `0.001` |
| Epochs | 10 in integrated/cloud drivers; signal functions default to 100. | 100 |
| Batch size | 256 | 256 in integrated drivers; 16 in edge drivers. |
| Target | Leaf parent-group class. | Binary emotion class. |

## Outputs and reproducibility

The scripts print classification accuracy, weighted F1, losses, and timing. Cloud scripts save full serialized encoder objects. The non-cluster integrated driver saves `model.pth`; the cluster driver saves `gcn_model.pth`, overwriting that filename across folds. Use fold-specific names when retaining results.

A `Train_GCN/model.pth` file is already tracked, but its training configuration and provenance are not documented. It should not be assumed to represent the thesis's final model. Full-object checkpoints also depend on matching Python class definitions and serialization settings; for portable experiments, save architecture/configuration metadata together with state dictionaries.

Before reporting results:

- Fit preprocessing and clustering within the training subjects of each fold. Current normalization is applied separately to partitions, and non-cluster cloud scripts normalize before reconstructing LOSO folds. Record whether any held-out-user calibration is intentionally transductive.
- Keep self-supervised training free of held-out subjects, and keep adaptation labels separate from final evaluation labels. Overlapping signal windows may require separation at the recording/trial level rather than by adjacent cache columns.
- Fix seeds and preserve subject/frame order. Use validation data for epoch selection: edge scripts currently report the best accuracy and best F1 observed over test epochs, potentially from different epochs.
- Retain every evaluation sample and report aggregate results across users. Verify that all modality/label IDs match before model training.
- Treat printed latency as unvalidated timing: the code measures batch forward calls without explicit CUDA synchronization, despite labeling them as one-window timings. FLOP estimates also require checking profiler support for graph operations.

The repository provides the algorithmic experiment components, not a complete Jetson deployment package or a verified reproduction of Chapter 5's final scores.

## Citation

If you use this code or the associated framework, please cite:

```bibtex
@inproceedings{sun2026cloudheavy,
  author    = {Sun, Junjiao and Miranda, J. and Portilla, Jorge and Otero, Andrés},
  title     = {From Cloud-Heavy to Edge-Ready: Self-supervised Transfer-efficient Emotion Recognition},
  booktitle = {2026 Design, Automation \& Test in Europe Conference (DATE)},
  year      = {2026},
  pages     = {1--3},
  doi       = {10.23919/DATE69613.2026.11539461}
}
```

Please also cite the relevant dataset publications when using their data.

## License

No repository-level license file is included. Contact the maintainers to clarify reuse and redistribution terms. Dataset access and reuse remain subject to the respective providers' terms.
