# ResHGNN: Sparse Residual Heterogeneous Graph Neural Networks for Efficient Insider Threat Detection

This repository provides the implementation, sample data, graph construction pipeline, model definitions, and experimental scripts for the under-review paper:

**ResHGNN: Sparse Residual Heterogeneous Graph Neural Networks for Efficient Insider Threat Detection**

## Overview

Insider threat detection is a critical cybersecurity task because malicious insider behaviours are often rare, concealed within normal daily activities, and difficult to distinguish from legitimate user behaviour. This project implements **ResHGNN**, a sparse residual heterogeneous graph neural network framework for efficient insider threat detection.

The framework models insider threat detection as a heterogeneous graph learning problem. It represents daily user activities as target nodes and incorporates organisational and behavioural relations, such as user identity and supervisory structure, as heterogeneous relational information. Residual learning is used to preserve the original user-day behavioural representation while allowing graph neural networks to capture relational signals from the heterogeneous graph.

The repository is designed to support reproducibility, ablation experiments, and performance comparison across different graph relations and GNN backbones.

## Main Features

* Construction of a heterogeneous graph from sampled insider-threat activity data.

* User-day node representation based on manually extracted behavioural features.

* Heterogeneous relations including:

  * user-day to supervisor relation;

  * user-day to same-user identity relation.

* Residual heterogeneous graph learning for insider threat detection.

* Support for multiple GNN backbones, including GCN, GAT, and GraphSAGE.

* K-fold validation during training.

* Early stopping for model selection.

* Performance reporting with standard classification metrics.

## Repository Structure

```text
.
├── data/
│   ├── data-total.csv
│   ├── insider_detection_heterogeneous_graph.pt
│   └── userlist.csv
├── 1_graph_construction.py
├── 2_No_sequence_detection_EndToEnd_k_fold.py
├── 3_No_sequence_process_perform_file.py
├── No_sequence_models_EndToEnd_K_fold.py
├── tool_EndToEnd_KFolds.py
├── early_stop_v1.py
└── README.md
```

## File Description

### data/

This directory contains the prepared sample data and graph data used by the experimental scripts.

* `data-total.csv`: processed user-day activity records and labels.

* `userlist.csv`: user information used to construct organisational relations.

* `insider_detection_heterogeneous_graph.pt`: saved PyTorch Geometric heterogeneous graph object.

### 1_graph_construction.py

Constructs the heterogeneous graph from the processed sample data.

This script creates:

* `UserDay` nodes representing daily user activity records;

* `supervisor` nodes representing supervisory entities;

* `user` nodes representing individual users;

* `UserDay -> supervisor` edges for supervisory relationships;

* `UserDay -> user` edges for same-user identity relationships;

* node labels and train-test masks for the insider threat detection task.

The constructed graph is saved as:

```text
./data/insider_detection_heterogeneous_graph.pt
```

### 2_No_sequence_detection_EndToEnd_k_fold.py

Runs the main ResHGNN detection experiment.

This script loads the heterogeneous graph, applies different relation settings and GNN backbones, trains the model with k-fold validation, applies early stopping, evaluates on the test set, and saves performance results.

The default experimental settings include:

* relation settings:

  * `All_relation`

  * `Supervision_relation`

  * `SameUser_relation`

* GNN models:

  * `GCN`

  * `GAT`

  * `GraphSAGE`

* residual learning enabled by default;

* early stopping based on validation loss;

* repeated experimental rounds for robust evaluation.

### 3_No_sequence_process_perform_file.py

Processes and summarises archived experimental results.

After running the detection experiments, this script can be used to organise performance records into structured CSV files for easier comparison and reporting. Users may need to modify the result directory path before running this script.

### No_sequence_models_EndToEnd_K_fold.py

Defines the model architectures used in the detection task.

The file includes:

* GNN backbone definitions;

* heterogeneous graph models with all relations;

* models using only the same-user relation;

* models using only the supervision relation;

* residual and non-residual variants;

* optional classifier modules.

### tool_EndToEnd_KFolds.py

Provides utility functions for evaluation, including metric calculation and result reporting.

### early_stop_v1.py

Implements the early stopping strategy used during model training.

## Dataset

The repository uses processed sample data derived from insider-threat activity records. The data are organised at the user-day level and include behavioural features, labels, user identifiers, date indices, and organisational information required for heterogeneous graph construction.

The raw enterprise activity logs are not included in this repository. The provided sample files are sufficient for running the graph construction and detection scripts included in this project.



## Quick Start

### 1. Install Dependencies

The code requires Python and common scientific computing and deep learning libraries. A typical environment includes:

```bash
pip install numpy pandas scikit-learn matplotlib torch torch-geometric
```

Please install the PyTorch and PyTorch Geometric versions that match your CUDA and operating system configuration.

### 2. Construct the Heterogeneous Graph

Run the graph construction script first:

```bash
python 1_graph_construction.py
```

This will create or overwrite:

```text
./data/insider_detection_heterogeneous_graph.pt
```

### 3. Run ResHGNN Detection Experiments

After graph construction, run:

```bash
python 2_No_sequence_detection_EndToEnd_k_fold.py
```

The script will train and evaluate the model under different relation settings and GNN backbones.

### 4. Summarise Experimental Results

After the detection experiments are complete, run:

```bash
python 3_No_sequence_process_perform_file.py
```

Before running this script, check and modify the root result path so that it points to the directory containing the output files generated in the previous step.

## Experimental Outputs

By default, the detection script saves results under a result directory such as:

```text
result_NoSeq_e5_emb/
```

The output files include:

* model checkpoints selected by early stopping;

* per-round performance files;

* total result CSV files;

* optional loss trends, depending on script settings.

The main reported metrics include:

* Accuracy;

* Precision;

* Recall;

* F1-score;

* AUC.

## Notes for Reproducibility

* Run `1_graph_construction.py` before the detection experiment if the graph file has not yet been generated.

* Ensure that `data-total.csv` and `userlist.csv` are stored in the `data/` directory.

* The current scripts use predefined train-test masks and experimental settings.

* File paths and result directories may need to be adjusted depending on the local environment.

* The default script performs repeated experiments, which may require substantial training time depending on hardware.

* GPU acceleration is used automatically when CUDA is available.



## Contact

For questions about the code, data preparation, or experimental settings, please open an issue in this repository.
