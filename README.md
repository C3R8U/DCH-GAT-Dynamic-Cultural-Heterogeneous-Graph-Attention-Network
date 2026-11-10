# DCH-GAT-Dynamic-Cultural-Heterogeneous-Graph-Attention-Network
DCH-GAT is a Python research project for analyzing short-video cultural dissemination through typed graph relations, cultural metadata, semantic paths, and daily network states. The learning system provides three task outputs: diffusion reach estimation, influential-user ranking, and dissemination peak-time estimation.

## 1. Installation

Run the following commands from the `DCHGAT_Study_Code` directory:

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest -q tests/
```

The package requires Python 3.10 or later. Its Python dependencies are declared in `pyproject.toml` and `requirements.txt`. The stored Python, operating-system, PyTorch, and device information is in `configs/execution_environment.json`.

## 2. Data and chronological organization

Five node classes define the cultural heterogeneous information network: **users**, **short videos**, **music genres**, **cultural symbols**, and **traditional instruments**. Relations represent user following, authorship, user-video interactions, and associations between videos and cultural attributes; forward and reverse relation types are stored separately.

| Partition | Video publication dates | Purpose |
|---|---|---|
| Training | 2024-01-01 to 2025-03-31 | Optimize model weights and target scaling. |
| Validation | 2025-04-01 to 2025-04-30 | Select checkpoints and training settings. |
| Test | 2025-05-01 to 2025-06-30 | Calculate final held-out metrics. |

`data/source/` holds the provided input tables, while `data/working/` holds the chronological analysis tables prepared by `scripts/prepare_data.py`. The graph model uses daily interaction snapshots for days 0 and 1 after publication and a seven-day observation window for the diffusion targets.

### Core graph relations

| Relation families | Interpretation |
|---|---|
| `follows`, `followed_by` | Directed user-to-user social links. |
| `published`, `published_by` | Connections between authors and their videos. |
| `interacts`, `interacted_by` | Timestamped user participation in videos. |
| `has_genre`, `genre_of` | Video-to-music-genre links. |
| `has_symbol`, `symbol_of` | Video-to-cultural-symbol links. |
| `has_instrument`, `instrument_of` | Video-to-traditional-instrument links. |

## 3. Model architecture

### 3.1 Node feature encoders

Each node type has a separate learnable feature projection into a shared 128-dimensional latent space. The structured video representation uses encoded categorical metadata, numerical descriptors, and hashed description text. Optional audiovisual descriptors are accepted through a validated media feature file.

The visual encoder uniformly samples eight frames per clip, resizes them to 224 by 224 pixels, and extracts ResNet-50 global-average-pooling features; averaging across frames produces a **2,048-dimensional** visual descriptor.

The audio encoder converts a mono 16 kHz waveform into a log-Mel spectrogram using an FFT size of 512, hop length 256, and 64 Mel bands. A bidirectional LSTM with 64 hidden units per direction produces a **128-dimensional** audio descriptor by concatenating the final forward and backward hidden states.

When audiovisual descriptors are supplied, they are concatenated with the structured video features before projection into the graph embedding space.

### 3.2 Heterogeneous relation attention

Three graph attention layers operate on the heterogeneous network. Each layer uses eight attention heads, giving 16 features per head for a 128-dimensional hidden state. Relation-specific source and destination attention vectors and message scaling parameters govern neighbor aggregation. Attention coefficients are normalized over incoming neighbors for each relation and head. Residual projection, ELU nonlinearity, layer normalization, and dropout complete each layer.

### 3.3 Semantic-path attention

Five two-step semantic paths organize shared users and cultural attributes:

- `Video -> User -> Video`
- `Video -> Genre -> Video`
- `Video -> Cultural Symbol -> Video`
- `Video -> Traditional Instrument -> Video`
- `User -> Video -> User`

Sparse path-based neighbor aggregation produces semantic embeddings. A tanh transformation and learned softmax weights combine the path representations for each target node.

### 3.4 Temporal aggregation and prediction heads

A Gated Recurrent Unit (GRU) cell updates node embeddings across two daily graph snapshots. Final video embeddings feed separate diffusion-reach and peak-day regression heads. Video and user embeddings feed a user-video ranking scorer that generates influence scores for candidate users.

## 4. Training configuration

| Setting | Value |
|---|---|
| Embedding dimension | `128` |
| Graph attention heads | `8` |
| Graph layers | `3` |
| Attention-head width | `16` |
| Dropout | `0.5` |
| Optimizer | `Adam` |
| Learning rate | `0.001` |
| Weight decay | `0.0005` |
| Ranking-pair batch size | `1024` |
| Maximum epochs | `60` |
| Early-stopping patience | `20` epochs |
| Gradient clipping maximum norm | `2.0` |
| Daily input snapshots | `2` |
| Diffusion outcome window | `7` days |
| Influence decay coefficient | `0.3` |
| Independent random seeds | `2021`, `2022`, `2023`, `2024`, `2025` |
| Checkpoint criterion | Lowest validation objective |

The reference workstation specification is Ubuntu 20.04 LTS, Intel Xeon Gold 6248R, NVIDIA A100 80 GB, and 256 GB of RAM. Device details for the saved experiment runs are recorded separately in `configs/execution_environment.json`.

### Training objective

The optimization loss combines three supervised terms:

```text
L_total = L_reach + 0.35 * L_peak + 0.25 * L_rank
L_reach = mean squared error of standardized log reach
L_peak  = Smooth L1 loss of peak day
L_rank  = binary cross-entropy with logits for sampled user-video pairs
```

The diffusion reach target is the number of distinct recorded participating users within the seven-day window. Reach is transformed with `log1p` and standardized using the training subset. Peak day is the earliest daily maximum in observed participation. Influence ranking uses time-decayed reachability within the user-follow network, selecting the top 20 candidates for each eligible video.

### Evaluation metrics

| Task | Metrics |
|---|---|
| Diffusion reach prediction | RMSE and MAE on standardized log reach, plus MAE in participant units. |
| Influential-user identification | Precision@20, Recall@20, and F1@20. |
| Peak-time prediction | MAE in days and coefficient of determination (R²). |
| Across-run statistics | Mean, sample standard deviation, and 95% confidence intervals. |
| Sample-level uncertainty | 1,000 bootstrap resamples of held-out video predictions. |
| Paired comparisons | Paired t-test and Wilcoxon signed-rank test. |

## 5. Running the project

### Prepare the chronological tables

```bash
python scripts/prepare_data.py
python scripts/audit_integrity.py
```

### Train the five independent runs and component ablations

```bash
python scripts/run_experiments.py --epochs 60 --seeds 2021 2022 2023 2024 2025 --ablations
```

The training runner creates per-seed checkpoints, per-epoch histories, validation and test predictions, baseline comparisons, and general visualization figures in `outputs/`.

### Aggregate metrics and statistical comparisons

```bash
python scripts/paired_tests.py
python scripts/summarize_results.py
python scripts/inspect_attention.py
python scripts/analyze_subgroups.py
```

### Run hyperparameter sensitivity experiments

```bash
python scripts/sensitivity.py --epochs 30 --seed 2021
```

The sensitivity grid covers embedding dimensions 32, 64, 128, and 256 and attention-head counts 2, 4, 8, and 16, with 128 dimensions and eight heads used as the primary configuration.

### Extract and attach local audiovisual features

Provide a CSV manifest containing the columns `video_id`, `video_path`, and `audio_path`, with one entry for each analysis video. The optional extraction and training commands are:

```bash
python scripts/extract_media.py --manifest media_manifest.csv --output data/working/media_features.npz --frames 8
python scripts/run_experiments.py --media-features data/working/media_features.npz --epochs 60 --seeds 2021 2022 2023 2024 2025 --ablations
```

The feature file stores aligned video IDs and separate visual and audio descriptor matrices. All video IDs in that file must match the order of `data/working/videos.csv`.

### Validate and list project files

```bash
python -m pytest -q tests/
python scripts/validate_project.py
python scripts/write_manifest.py
```

## 6. Experiment outputs

Training records are saved as `history.csv`, `metrics.json`, `best_model.pt`, `validation_predictions.csv`, and `test_predictions.csv` under each experiment folder. The aggregated summaries are in `outputs/seed_metrics.csv`, `outputs/seed_summary.json`, `outputs/ablation_metrics.csv`, and `outputs/RESULTS.md`. Model topology, node statistics, learning curves, scatterplots, residuals, attention weights, rank overlap, and controlled-perturbation plots are stored in `outputs/figures/`.

![DCH-GAT network schema](outputs/figures/network_schema.png)

![DCH-GAT model architecture](outputs/figures/architecture.png)

## 7. File reference

Each entry below describes one existing file in the research package. Repeated experiment directories share the same file conventions and are listed individually for direct lookup.

- `README.md` — Introduces the project, configuration, execution steps, and complete file inventory.

### Project configuration and documentation

- `Makefile` — Provides short command targets for installation, preparation, testing, training, aggregation, attention inspection, and sensitivity analysis.
- `configs/data_inventory.json` — Stores file checksums, source counts, selected date boundaries, and chronological split counts.
- `configs/execution_environment.json` — Records the execution environment, fixed random seeds, model settings, and reference compute profile.
- `docs/DATA_PROVENANCE.md` — Documents the input archive, data selection, available observations, and dataset interpretation.
- `docs/EXPERIMENT_SPECIFICATION.md` — Describes the graph schema, target calculations, training objective, evaluation protocol, and media feature format.
- `pyproject.toml` — Declares the Python package, build backend, minimum interpreter version, and runtime dependencies.
- `requirements.txt` — Lists the Python libraries used by the model, data pipeline, plotting utilities, and tests.

### Input and prepared data

- `data/source/cascades.csv` — Provides the original input table: records timestamped participant events in video diffusion cascades.
- `data/source/edges_user_follow.csv` — Provides the original input table: stores directed user-follow relationships.
- `data/source/edges_user_interactions.csv` — Provides the original input table: stores timestamped user-video engagement relationships.
- `data/source/music.csv` — Provides the original input table: stores music metadata and descriptors indexed by music identifiers.
- `data/source/users.csv` — Provides the original input table: stores user identifiers and user-level attributes.
- `data/source/video_music.csv` — Provides the original input table: maps videos to associated music identifiers.
- `data/source/video_split.csv` — Provides the original input table: assigns video identifiers to the train, validation, or test partition.
- `data/source/video_tags.csv` — Provides the original input table: associates videos with cultural labels and symbols.
- `data/source/videos.csv` — Provides the original input table: stores video identifiers, publication timestamps, authors, and video-level descriptors.
- `data/working/cascades.csv` — Provides the selected chronological working table: records timestamped participant events in video diffusion cascades.
- `data/working/data_counts.json` — Summarizes record counts and chronological selection counts for the working tables.
- `data/working/edges_user_follow.csv` — Provides the selected chronological working table: stores directed user-follow relationships.
- `data/working/edges_user_interactions.csv` — Provides the selected chronological working table: stores timestamped user-video engagement relationships.
- `data/working/music.csv` — Provides the selected chronological working table: stores music metadata and descriptors indexed by music identifiers.
- `data/working/users.csv` — Provides the selected chronological working table: stores user identifiers and user-level attributes.
- `data/working/video_music.csv` — Provides the selected chronological working table: maps videos to associated music identifiers.
- `data/working/video_split.csv` — Provides the selected chronological working table: assigns video identifiers to the train, validation, or test partition.
- `data/working/video_tags.csv` — Provides the selected chronological working table: associates videos with cultural labels and symbols.
- `data/working/videos.csv` — Provides the selected chronological working table: stores video identifiers, publication timestamps, authors, and video-level descriptors.

### Python modules and experiment scripts

- `scripts/analyze_subgroups.py` — Computes held-out subgroup metrics and controlled metadata-perturbation diagnostics.
- `scripts/audit_integrity.py` — Checks date boundaries, table references, observation windows, and graph data consistency.
- `scripts/extract_media.py` — Reads a local audiovisual manifest and exports ResNet-50 and BiLSTM descriptors to an NPZ file.
- `scripts/inspect_attention.py` — Exports learned semantic-path attention weights and creates attention visualizations.
- `scripts/paired_tests.py` — Calculates paired held-out differences, significance tests, and confidence-interval graphics.
- `scripts/prepare_data.py` — Builds the chronologically filtered working tables and the train, validation, and test split assignments.
- `scripts/run_experiments.py` — Runs DCH-GAT training, baseline models, structural ablations, and experiment plots.
- `scripts/sensitivity.py` — Evaluates selected hidden dimensions and attention-head counts with saved metrics and charts.
- `scripts/summarize_results.py` — Aggregates independent-run metrics, bootstrap intervals, ablation outcomes, and the results report.
- `scripts/validate_project.py` — Checks required project outputs, trained runs, data counts, and figure inventory.
- `scripts/write_manifest.py` — Writes an inventory of project files, model runs, and ablation outputs.
- `src/dchgat/__init__.py` — Exposes the package namespace and its public model/data interfaces.
- `src/dchgat/data.py` — Loads structured CSV tables, creates typed graph views, constructs features, and calculates diffusion targets.
- `src/dchgat/media.py` — Defines the ResNet-50 visual encoder, log-Mel BiLSTM audio encoder, and media-loading utilities.
- `src/dchgat/model.py` — Implements relation attention, semantic-path aggregation, temporal GRU states, and prediction heads.
- `src/dchgat/plots.py` — Generates network diagrams, data summaries, training curves, predictions, and evaluation figures.
- `src/dchgat/subgroups.py` — Evaluates group-specific prediction errors and feature-shuffle experiments.
- `src/dchgat/train.py` — Defines training configuration, optimizer steps, ranking supervision, model evaluation, and tabular baselines.
- `tests/test_pipeline.py` — Tests preprocessing, graph construction, model execution, and core experimental behavior.

### Training and ablation directories

- `outputs/ablation_no_attention/best_model.pt` — Stores the selected model weights, saved configuration, and best validation epoch.
- `outputs/ablation_no_attention/config.json` — Stores the hyperparameters and enabled model components for this run.
- `outputs/ablation_no_attention/history.csv` — Records per-epoch training loss, validation loss, loss components, and runtime information.
- `outputs/ablation_no_attention/metrics.json` — Stores held-out and validation evaluation metrics and run metadata.
- `outputs/ablation_no_attention/test_predictions.csv` — Stores video-level held-out targets, predictions, and ranking scores.
- `outputs/ablation_no_attention/validation_predictions.csv` — Stores video-level validation targets, predictions, and ranking scores.
- `outputs/ablation_no_content/best_model.pt` — Stores the selected model weights, saved configuration, and best validation epoch.
- `outputs/ablation_no_content/config.json` — Stores the hyperparameters and enabled model components for this run.
- `outputs/ablation_no_content/history.csv` — Records per-epoch training loss, validation loss, loss components, and runtime information.
- `outputs/ablation_no_content/metrics.json` — Stores held-out and validation evaluation metrics and run metadata.
- `outputs/ablation_no_content/test_predictions.csv` — Stores video-level held-out targets, predictions, and ranking scores.
- `outputs/ablation_no_content/validation_predictions.csv` — Stores video-level validation targets, predictions, and ranking scores.
- `outputs/ablation_no_culture_edges/best_model.pt` — Stores the selected model weights, saved configuration, and best validation epoch.
- `outputs/ablation_no_culture_edges/config.json` — Stores the hyperparameters and enabled model components for this run.
- `outputs/ablation_no_culture_edges/history.csv` — Records per-epoch training loss, validation loss, loss components, and runtime information.
- `outputs/ablation_no_culture_edges/metrics.json` — Stores held-out and validation evaluation metrics and run metadata.
- `outputs/ablation_no_culture_edges/test_predictions.csv` — Stores video-level held-out targets, predictions, and ranking scores.
- `outputs/ablation_no_culture_edges/validation_predictions.csv` — Stores video-level validation targets, predictions, and ranking scores.
- `outputs/ablation_no_meta/best_model.pt` — Stores the selected model weights, saved configuration, and best validation epoch.
- `outputs/ablation_no_meta/config.json` — Stores the hyperparameters and enabled model components for this run.
- `outputs/ablation_no_meta/history.csv` — Records per-epoch training loss, validation loss, loss components, and runtime information.
- `outputs/ablation_no_meta/metrics.json` — Stores held-out and validation evaluation metrics and run metadata.
- `outputs/ablation_no_meta/test_predictions.csv` — Stores video-level held-out targets, predictions, and ranking scores.
- `outputs/ablation_no_meta/validation_predictions.csv` — Stores video-level validation targets, predictions, and ranking scores.
- `outputs/ablation_no_temporal/best_model.pt` — Stores the selected model weights, saved configuration, and best validation epoch.
- `outputs/ablation_no_temporal/config.json` — Stores the hyperparameters and enabled model components for this run.
- `outputs/ablation_no_temporal/history.csv` — Records per-epoch training loss, validation loss, loss components, and runtime information.
- `outputs/ablation_no_temporal/metrics.json` — Stores held-out and validation evaluation metrics and run metadata.
- `outputs/ablation_no_temporal/test_predictions.csv` — Stores video-level held-out targets, predictions, and ranking scores.
- `outputs/ablation_no_temporal/validation_predictions.csv` — Stores video-level validation targets, predictions, and ranking scores.
- `outputs/full_seed_2021/best_model.pt` — Stores the selected model weights, saved configuration, and best validation epoch.
- `outputs/full_seed_2021/config.json` — Stores the hyperparameters and enabled model components for this run.
- `outputs/full_seed_2021/history.csv` — Records per-epoch training loss, validation loss, loss components, and runtime information.
- `outputs/full_seed_2021/metrics.json` — Stores held-out and validation evaluation metrics and run metadata.
- `outputs/full_seed_2021/test_predictions.csv` — Stores video-level held-out targets, predictions, and ranking scores.
- `outputs/full_seed_2021/validation_predictions.csv` — Stores video-level validation targets, predictions, and ranking scores.
- `outputs/full_seed_2022/best_model.pt` — Stores the selected model weights, saved configuration, and best validation epoch.
- `outputs/full_seed_2022/config.json` — Stores the hyperparameters and enabled model components for this run.
- `outputs/full_seed_2022/history.csv` — Records per-epoch training loss, validation loss, loss components, and runtime information.
- `outputs/full_seed_2022/metrics.json` — Stores held-out and validation evaluation metrics and run metadata.
- `outputs/full_seed_2022/test_predictions.csv` — Stores video-level held-out targets, predictions, and ranking scores.
- `outputs/full_seed_2022/validation_predictions.csv` — Stores video-level validation targets, predictions, and ranking scores.
- `outputs/full_seed_2023/best_model.pt` — Stores the selected model weights, saved configuration, and best validation epoch.
- `outputs/full_seed_2023/config.json` — Stores the hyperparameters and enabled model components for this run.
- `outputs/full_seed_2023/history.csv` — Records per-epoch training loss, validation loss, loss components, and runtime information.
- `outputs/full_seed_2023/metrics.json` — Stores held-out and validation evaluation metrics and run metadata.
- `outputs/full_seed_2023/test_predictions.csv` — Stores video-level held-out targets, predictions, and ranking scores.
- `outputs/full_seed_2023/validation_predictions.csv` — Stores video-level validation targets, predictions, and ranking scores.
- `outputs/full_seed_2024/best_model.pt` — Stores the selected model weights, saved configuration, and best validation epoch.
- `outputs/full_seed_2024/config.json` — Stores the hyperparameters and enabled model components for this run.
- `outputs/full_seed_2024/history.csv` — Records per-epoch training loss, validation loss, loss components, and runtime information.
- `outputs/full_seed_2024/metrics.json` — Stores held-out and validation evaluation metrics and run metadata.
- `outputs/full_seed_2024/test_predictions.csv` — Stores video-level held-out targets, predictions, and ranking scores.
- `outputs/full_seed_2024/validation_predictions.csv` — Stores video-level validation targets, predictions, and ranking scores.
- `outputs/full_seed_2025/best_model.pt` — Stores the selected model weights, saved configuration, and best validation epoch.
- `outputs/full_seed_2025/config.json` — Stores the hyperparameters and enabled model components for this run.
- `outputs/full_seed_2025/history.csv` — Records per-epoch training loss, validation loss, loss components, and runtime information.
- `outputs/full_seed_2025/metrics.json` — Stores held-out and validation evaluation metrics and run metadata.
- `outputs/full_seed_2025/test_predictions.csv` — Stores video-level held-out targets, predictions, and ranking scores.
- `outputs/full_seed_2025/validation_predictions.csv` — Stores video-level validation targets, predictions, and ranking scores.

### Experiment summaries, logs, and diagnostic tables

- `outputs/RESULTS.md` — Contains aggregated run measurements, baseline evaluations, ablations, and statistical comparisons.
- `outputs/ablation_metrics.csv` — Tabulates held-out metrics for individual component-ablation configurations.
- `outputs/ablation_pid.txt` — Stores the recorded process identifier for the ablation training job.
- `outputs/ablation_training_console.log` — Contains epoch-by-epoch console output for the ablation training runs.
- `outputs/attention_console.log` — Records the console output from semantic-path attention analysis.
- `outputs/baselines/baseline_metrics.json` — Stores the evaluation metrics calculated for non-graph comparison models.
- `outputs/baselines/baseline_test_predictions.csv` — Stores held-out predictions from the tabular baseline models.
- `outputs/bootstrap_1000_ci.json` — Stores 95% bootstrap confidence limits from 1,000 held-out video resamples.
- `outputs/file_manifest.json` — Lists package files and model-run artifacts with file metadata and checksums.
- `outputs/package_validation.json` — Records the results of automated project-package validation checks.
- `outputs/paired_comparisons.json` — Stores paired error and ranking differences with significance-test statistics.
- `outputs/paired_test_predictions.csv` — Combines per-video model and baseline predictions for paired analyses.
- `outputs/paired_tests_console.log` — Records execution messages from the paired statistical comparison script.
- `outputs/preprocessing/audit.json` — Summarizes node counts, graph properties, split assignments, and computed target statistics.
- `outputs/preprocessing/data_quality.json` — Reports source-table integrity checks and observation-window statistics.
- `outputs/preprocessing/targets.csv` — Stores per-video reach, peak-time, and standardized training-target variables.
- `outputs/preprocessing/verified_splits.csv` — Stores video IDs, publication times, and confirmed chronological partition labels.
- `outputs/primary_exit_code.txt` — Stores the completion status code from the main training job.
- `outputs/primary_pid.txt` — Stores the recorded process identifier for the main training job.
- `outputs/primary_training_console.log` — Contains console training output for the five independent main runs.
- `outputs/seed_metrics.csv` — Summarizes test metrics and checkpoint epochs for each independent random seed.
- `outputs/seed_summary.json` — Stores cross-seed metric means, standard deviations, and confidence intervals.
- `outputs/semantic_path_weights.csv` — Stores learned semantic-route attention weights exported from a trained model.
- `outputs/subgroups/creator_coldstart.csv` — Summarizes held-out metrics for creator-history threshold groups.
- `outputs/subgroups/group_metrics.csv` — Stores evaluation metrics by genre, instrument, region, quarter, and creator-history group.
- `outputs/subgroups/metadata_perturbation.csv` — Records metric changes after repeated within-month cultural-metadata shuffles.
- `outputs/subgroups/reference_test_metrics.json` — Stores reference metrics used for the metadata-perturbation calculations.
- `outputs/subgroups/test_annotated.csv` — Contains held-out predictions enriched with video attributes and creator-history groups.
- `outputs/subgroups_console.log` — Records execution messages from the subgroup and metadata-perturbation analyses.

### Visualization figures

- `outputs/figures/architecture.png` — Raster model diagram showing feature encoding, heterogeneous attention, semantic aggregation, and temporal learning.
- `outputs/figures/architecture.svg` — Vector version of the DCH-GAT model architecture diagram.
- `outputs/figures/component_ablations.png` — Compares the held-out effects of omitting graph model components.
- `outputs/figures/early_event_window.png` — Visualizes the distribution of recorded events across the early observation window.
- `outputs/figures/follow_degree.png` — Displays follower-network degree statistics.
- `outputs/figures/genre_distribution.png` — Summarizes the music-genre distribution in the selected video cohort.
- `outputs/figures/model_comparison.png` — Compares selected evaluation metrics across the main model and baselines.
- `outputs/figures/network_schema.png` — Raster diagram of the typed nodes and relationships in the cultural graph.
- `outputs/figures/network_schema.svg` — Vector version of the cultural graph schema.
- `outputs/figures/node_inventory.png` — Displays counts of nodes by entity type.
- `outputs/figures/paired_baseline_differences.png` — Shows paired model-minus-baseline differences with uncertainty intervals.
- `outputs/figures/path_attention_heatmap.png` — Displays learned semantic-path attention weights as a heatmap.
- `outputs/figures/path_attention_mean.png` — Summarizes average learned weights for the semantic routes.
- `outputs/figures/peak_confusion.png` — Visualizes the distribution of predicted and observed peak-day categories.
- `outputs/figures/peak_prediction.png` — Plots predicted peak-time values against observed peak days.
- `outputs/figures/rank_overlap.png` — Displays agreement between predicted and reference influential-user rankings.
- `outputs/figures/sample_heterogeneous_graph.png` — Visualizes a selected subgraph with differently typed entity nodes.
- `outputs/figures/scope_residuals.png` — Plots reach-prediction residuals for error inspection.
- `outputs/figures/scope_scatter.png` — Compares predicted and observed standardized reach targets.
- `outputs/figures/seed_variation.png` — Shows metric variation across the five random seeds.
- `outputs/figures/subgroup_perturbation.png` — Displays subgroup error profiles and the effect of shuffled cultural metadata.
- `outputs/figures/target_distributions.png` — Visualizes the modeled reach and peak-time target distributions.
- `outputs/figures/temporal_split.png` — Shows the chronological boundaries of the training, validation, and test sets.
- `outputs/figures/training_curves.png` — Displays epoch-level training and validation loss curves.
- `outputs/figures/validation_components.png` — Shows the training objective components across validation histories.
