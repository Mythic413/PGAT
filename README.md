# Dataset & Preprocessing Pipeline

## Dataset Used
* **`higgs-social_network.edgelist.gz`**: This is the underlying structural graph representing the "follower/following" relationships.
* **`higgs-retweet_network.edgelist.gz`**: It tracks who retweeted whom, providing the specific pathways through which the information successfully diffused.
* **`higgs-activity_time.txt.gz`**: Tracking the exact timestamps of interactions, which is crucial for determining the sequence of activations in your diffusion models.

**Original Higgs:**
* **Nodes:** 456,626
* **Edges:** 14,855,842

---

## 1. Preprocessing

### 1.1. STEP 2: EXTRACT_SUBGRAPH FILE
#### **Try 1:**
* **What I did:** Loaded undirected graph ---> Found largest connected component (CC) ----> Ran BFS from hub node (1503) ----> Extracted and saved subgraph.
* **What I got:**
  * **Original:** 456,626 nodes | 14,855,842 edges
  * **Largest CC:** 456,290 nodes | 14,855,466 edges
  * **Hub Degree (Node 1503):** 51,388
  * **Final Subgraph:** 5,000 nodes | 121,844 edges
  * **Avg Degree:** 48.74 | **Density:** 0.004875
* **Problem: BFS Hairball Bias**
  * Unrealistic "hairball."
  * Destroys the natural scale-free network structure.
  * Ignores peripheral communities.
  * Renders influence diffusion models inaccurate.

#### **Try 2:**
* **What I did:**
  * Replaced BFS with Canonical Forest Fire Sampling.
  * Used Geometric burning to prevent hub explosions.
  * Implemented a restart mechanism to guarantee 5,000 nodes.
  * Tuned parameters: Forward=0.60, Backward=0.15.
* **What I got:**
  * **Original Higgs:** Nodes = 456,626 | Edges = 14,855,842 | Average Degree = 65.07
  * **Final Subgraph:** Nodes = 5,000 | Edges = 186,524 | Average Degree = 74.61
* **Attempts:**
  * Initial Tuning Attempt (F: 0.70, B: 0.20): 5,000 nodes | 236,715 edges | Avg Degree: 94.69 (~45% inflation).
  * Final Subgraph (F: 0.60, B: 0.15): 5,000 nodes | 186,524 edges | Avg Degree: 74.61 (~14.6% inflation).

---

### 1.2 STEP 3: STRUCTURAL_FEATURES FILE
#### **Try 1:**
* **What I did:** Computed global centralities (Betweenness, Closeness, Eigenvector, PageRank) + local features -> Applied MinMax normalization across all 5000 nodes.
* **What I got:**
  * Data Leakage.
  * Global centralities indirectly exposed validation/test topology to training nodes.
  * Normalization violated inductive assumptions by fitting test set extrema.
  * Mean Degree: 48.74 | Max: 5001
  * Mean Clustering: 0.4018
  * Mean Betweenness: 1.98e-04 | Mean Closeness: 0.5022
* **Structural Features Used (Try 1 - Global + Local, MinMax Normalized):**
  * Degree, Betweenness Centrality, Closeness Centrality, Eigenvector Centrality, PageRank, Clustering Coefficient

#### **Try 2:**
* **What I did:** Adopted pure GNN approach -> Dropped all global centralities -> Dropped MinMax normalization completely (deferred to Step 7) -> Computed strictly local, inductive features (in_degree, out_degree, degree, clustering).
* **What I got:** Successfully eliminated both global centrality leakage and train-test normalization leakage.
  * Mean In-Degree / Out-Degree: 37.3048
  * Mean Total Degree: 74.6096
  * Mean Clustering: 0.245348
  * Final Feature Matrix Shape: (5000, 5)
* **Structural Features Used (Try 2 - Strictly Local, Unnormalized):**
  * in_degree, out_degree, degree, clustering

---

### 1.3 STEP 4: BEHAVIORAL_FEATURES
#### **Try 1:**
* **What I did:** Extracted basic behavioral features from assumed-sorted activity logs. Applied MinMax normalization across all nodes.
* **What I got:** Normalization leakage and timestamp bugs (incorrect spans).
  * Processed Activities: 11,438
  * Feature Matrix: (5000, 6)
  * Activity Count: Mean = 2.28 | Max = 120
  * Activity Span: Mean = 30,978 | Max = 571,102
  * Ratios (Mean): RT = 0.418 | MT = 0.247 | RE = 0.036
* **Behavioral Features Used:** node, activity_count, retweet_ratio, mention_ratio, reply_ratio, activity_span

#### **Try 2:**
* **What I did:** Removed normalization. Fixed timestamp ordering using min()/max(). Added `is_active` binary indicator to handle behavioral sparsity explicitly.
* **What I got:** Fixed timestamp bugs, eliminated leakage, and mapped inactivity perfectly.
  * Processed Activities: 12,469
  * Feature Matrix: (5000, 7)
  * Activity Count: Mean = 2.49 | Median = 1 | Max = 656
  * Activity Span: Mean = 32,606 | Min = 0 | Max = 593,957
  * Ratios (Mean): RT = 0.342 | MT = 0.248 | RE = 0.036
  * is_active: Mean = 0.6272
* **Behavioral Features Used:** node, activity_count, retweet_ratio, mention_ratio, reply_ratio, activity_span, is_active

---

### 1.4 STEP 5: EDGE_PROBABILITIES
#### **Try 1:**
* **What I did:** Custom formula that combined structural features, behavioral features, and historical retweet interactions.
* **How I calculated:** * Structural component: `P_struct = degree(v)`
  * Behavioral component: `P_behav = 0.5 * activity_count(v) + 0.5 * retweet_ratio(v)`
  * Base probability: `p = 0.5 * P_struct + 0.5 * P_behav`
  * Retweet Boost: Multiplied p by 1.5 if the edge (u, v) existed in the retweet dataset.
  * IC Clipping: Bounded the final probability strictly between [0.01, 0.5].
* **Problems I got:** Predicting labels explicitly engineered from the inputs.
  * Circular Leakage: Target labels for the IC/LT models became direct mathematical functions of the input features.
  * GNNs could achieve high performance by completely ignoring the graph topology and simply reverse-engineering the formula.
* **Results:**
  * Edges Processed: 121,844 | Matched Retweet Events: 1,108
  * Boosted Edges: 751 | Non-Boosted Edges: 121,093
  * Probability Mean: 0.1448 | Median: 0.1198
  * Probability Min: 0.01 | Max: 0.50
  * Final Matrix Shape: (121844, 3)

#### **Try 2:**
* **What I did:** Scrapped the custom feature-based probabilities and adopted the Canonical Weighted Cascade model.
* **How I calculated:** Used the feature-independent formula: `p(u,v) = 1 / max(in_degree(v), 1)`.
* **Problems solved:** Completely eliminated circular leakage. The probabilities are now purely structural and literature-supported, forcing the GNN to genuinely learn diffusion dynamics through message passing.
* **Results:**
  * Edges Processed: 186,524
  * Probability Mean: 0.0263
  * Probability Median: 0.0092
  * Maximum Probability: 1.0
  * Output Shape: (186524, 3)

---

### 1.5 STEP 5A: IC_LABELS
#### **Try 1:**
* **What I did:** Generated (IC) target labels for 1,000 nodes using degree-stratified quartile sampling (250 nodes per quartile). Introduced randomized dynamic scale factors, enforced hard probability boundary clipping, and added a variance penalty to calculate a custom robust spread metric.
* **How I calculated:** * Executed 200 Monte Carlo runs per node.
  * Edge Probability Formula: `p = max(0.001, min(0.25, p_edge * IC_SCALE))` where `IC_SCALE` was randomly selected per run from {0.15, 0.20, 0.25, 0.30}.
  * Robust Objective Formula: `ic_robust = ic_mean - 0.1 * ic_std`
* **Problems I got:**
  * Dynamic scaling changed graph physics on every run.
  * Clipping destroyed Weighted Cascade properties.
  * Created non-standard objectives, causing reviewer pushback regarding mathematical inconsistency.
* **Numerical Results:**
  * Graph Setup: 5,000 nodes | 121,844 edges | 1,000 labeled nodes
  * ic_mean: Mean = 37.2656 | Std = 29.2795 | Max = 143.0500
  * ic_robust: Mean = 29.9790 | Std = 26.8012 | Max = 132.1394
  * Robustness Penalty lambda = 0.1: Mean = 7.2866 | Std = 2.8254 | Max = 11.8359

#### **Try 2:**
* **What I did:** Refactored the script to match standard graph physics. Eliminated random scaling, removed probability clipping, dropped the robust objective variance penalty, and saved a unified node universe (`labeled_nodes.csv`) to share across models.
* **How I calculated:**
  * Executed 200 standard Monte Carlo sweeps per node using unmodified edge weights.
  * Activation Formula: `If Uniform(0, 1) < p(u,v)`, activate node v.
  * Expected Spread Formula: `ic_mean = {1/M}*sum_{m=1}^{M} |A_m|` (Where M = 200 and A_m is the set of all activated nodes in simulation m).
* **Problems solved:**
  * Restored mathematical consistency, enabling an objective.
  * Synchronized the node sampling universe across IC and LT pipelines to ensure fair shared supervision.
* **Numerical Results:**
  * Graph Setup: 5,000 nodes | 186,524 edges | 1,000 labeled nodes
  * Mean spread: 10.82
  * Max spread: 114.74
  * Degree correlation: 0.398

---

### 1.6 STEP 5B: LT_LABELS
#### **Try 1:**
* **What I did:** Generated Linear Threshold (LT) target labels across 1,000 degree-stratified nodes using full graph scanning O(V) per propagation loop, explicit edge weight renormalization, and a safety-penalized robust spread metric.
* **How I calculated:** * Weight Renormalization Formula: `w(u,v) = p(u,v) / sum_{k \in N^{in}(v)} p(k,v)`
  * Activation Condition: Activate node v if: `sum_{u in A} w(u,v) >= theta_v` where `theta_v` is `Uniform(0, 1)`
  * Robust Spread Formula: `lt_robust = lt_mean - 0.1 * lt_std`
* **Problems I got & numerical results:**
  * Algorithmic Bottleneck: Repeatedly scanned all 5,000 nodes at every step.
  * Redundant Math: Weighted Cascade weights already satisfied LT constraints (where sum w <= 1).
  * Sampling Mismatch: Sampled an independent node set distinct from the IC pipeline.
  * Graph: 5,000 nodes | 121,844 edges | 1,000 independent labels
  * lt_mean: Mean = 10.07 | Median = 6.17 | Max = 155.47
  * Centrality Correlations: Degree = 0.7186 | PageRank = 0.4881 | Clustering = -0.5322
  * Columns generated: node, lt_mean, lt_std, lt_robust

#### **Try 2:**
* **What I did:** Optimized diffusion using a localized frontier accumulator. Reused the Step 5 Weighted Cascade edge probabilities directly without extra normalization, loaded a synchronized node set (`labeled_nodes.csv`), and removed the robust penalty.
* **How I calculated:**
  * Weight Rule: Reused unmodified Step 5 weights directly: `w(u,v) = p(u,v) = 1 / max(in_degree(v), 1)`
  * Frontier Accumulator Optimization: Incremented cumulative threshold weights strictly for active node successors: `influence[v] = influence[v] + w(u,v)`
  * Activation Condition: Activated node v immediately when: `influence[v] >= theta_v` where `theta_v` belongs to `Uniform(0, 1)`
  * Expected Spread Formula: `lt_mean = {1/M}*sum_{m=1}^{M} |A_m|` where M = 200 Monte Carlo runs
* **Problems solved & numerical results:**
  * Eliminated O(V) tracking complexity, removed mathematical redundancy, and established identical shared node supervision with IC.
  * Graph: 5,000 nodes | 186,524 edges | 1,000 unified labels
  * lt_mean: Mean spread = 14.71 | Max spread = 177.71
  * Degree Correlation: 0.3795
  * Columns generated: node, lt_mean, lt_std

---

### 1.7 STEP 7: Build_DATASET
#### **Try 1:**
* **What I did:** Preprocessed the `higgs_5000.edgelist` graph network into a PyTorch Geometric (PyG) dataset format. Combined 6 structural and 5 behavioral node features into a single matrix. Extracted edge probability weights and target cascade spread labels for training preparation.
* **How I did:**
  * Stored edge indexes using standard PyG mapping keys (`edge_index`, `edge_attr`).
  * Created training, validation, and testing masks using a simple randomized percentage split function.
* **Problems I faced in try 1:**
  * Edge Desynchronization: Edge indices and attributes became misaligned during parsing.
  * Missing Scaling: Raw feature values were passed directly without standardization, risking poor convergence.
  * Feature Misalignment: Assumed arbitrary node order match across independent CSV arrays.
  * Random Splits: Standard random split created significant distribution shift risks across topologies.
  * Missing Label Checks: Implied flawless parity between independent IC and LT data collections without verification.
* **Numerical results:**
  * Unified PyG Object generated with standard dimensions (contained unaligned feature shapes across 11 parameters).
  * Node split values: Random distribution across a 70/15/15 ratio.

#### **Try 2:**
* **What I did:** Alignment verification, edge tracking, and explicit data safety blocks.
* **Formulae used:**
  * Standardization Scaling: `z = (x - mu) / sigma`
* **How I did:**
  * Cleaned feature tracking to parse exactly 9 stable metrics (4 structural, 5 behavioral).
  * Built `edge_index` and `edge_attr` synchronously inside a unified processing loop.
  * Implemented a `StandardScaler` fitted exclusively on training masks.
  * Applied a degree-stratified partitioning methodology to guarantee structural balance across target labels.
* **Problems solved:**
  * Eradicated feature leakage completely via train-only fitted standard scaling.
  * Resolved edge attribute mismatch bugs by enforcing alignment assertions.
  * Erased distribution shifts by transitioning to degree-stratified data splits.
  * Guaranteed data integrity by checking label consistency for both IC and LT models.
* **Numerical results:**
  * PyG Object Structure: `x`: [5000, 9] (9 valid feature dimensions)
  * `edge_index`: [2, 186524]
  * `edge_attr`: [186524, 1]
  * `y_ic` / `y_lt`: 5000
  * Stratified Splits Output: Train Split: 700 nodes | Validation Split: 148 nodes | Test Split: 152 nodes

---
---

## 2. TRAINING

### 1.1 GCN-IC AND LT
#### **Try 1:**
* **NN Structure:** 2-Layer GCN (Input -> GCNConv -> ReLU -> Dropout -> GCNConv -> Output). Hidden Dimension: 32.
* **Technique Used:** Graph Convolutional Network (GCN) for Node Regression (IC Influence Estimation).
* **Learning Rate (LR):** 1e-3 | **Scheduler:** None | **Optimizer:** AdamW (Weight Decay: 1e-4) | **Epochs:** 1000 (Patience: 100).
* **Metrics: GCN-IC**
  * MAE: 23.407791 | RMSE: 34.528098 | R²: -0.370788 | Spearman: 0.321305 | NDCG@10: 0.577843 | Precision@10: 0.200000
* **GCN LT EXPERIMENT RESULTS:**
  * Best Epoch: 1000 | Best Validation MAE: 7.060779
  * Test Metrics: MAE: 6.842562 | RMSE: 11.170663 | R²: -0.179024 | Spearman: 0.499004 | NDCG@10: 0.336899 | Precision@10: 0.100000
* **Problems:**
  * Oversmoothing: Pure GCN forgot hub information.
  * Gradient Starvation: Huber loss (delta=0.1) was unsuitable for targets ranging from 1–115.
  * Dead ReLU: No stabilization mechanisms were used.
  * Weak Decoder: Output was mapped directly to 1 using GCNConv.

#### **Try 2:**
* **NN Structure:** 2-Layer GCN (Input → GCNConv → LayerNorm → ReLU → Dropout → GCNConv → Squeeze). Hidden: 32, Dropout: 0.10.
* **Technique Used:** GCN for Node Regression (IC Influence Estimation) with Masked Loss computation.
* **LR:** 1e-3 (Minimum LR: 1e-5) | **Scheduler:** ReduceLROnPlateau (Factor: 0.5, Patience: 20, Mode: Min).
* **Optimizer:** AdamW (Weight Decay: 1e-4, Gradient Clip: 1.0). Loss: MSELoss.
* **Epochs:** 500 (Early Stopping Patience: 100). Hit best validation at Epoch 499.

**GCN IC EXPERIMENT RESULTS:**
* Best Epoch: 499 | Best Validation MAE: 7.223066
* Test Metrics: MAE: 7.7849 | RMSE: 13.7476 | R²: 0.1311 | Spearman: 0.4474 | KendallTau: 0.3277
* Ranking: Precision@10: 0.3 | Precision@20: 0.4 | Recall@10: 0.3 | Recall@20: 0.4 | NDCG@10: 0.4512 | NDCG@20: 0.4800 | Top10Overlap: 3 | Top20Overlap: 8
* Elite Diagnostics: Top1_MAE: 90.26 | Top5_MAE: 56.49 | Top10_MAE: 39.57 | Top1_MSE: 8148.42 | P95_MAE: 45.35

**GCN LT EXPERIMENT RESULTS:**
* Best Epoch: 392 | Best Validation MAE: 11.102865
* Test Metrics: MAE: 11.3632 | RMSE: 20.6048 | R²: 0.0866 | Spearman: 0.3758 | KendallTau: 0.2731
* Ranking: Precision@10: 0.1 | Precision@20: 0.3 | Recall@10: 0.1 | Recall@20: 0.3 | NDCG@10: 0.3499 | NDCG@20: 0.4151 | Top10Overlap: 1 | Top20Overlap: 6
* Elite Diagnostics: Top1_MAE: 147.38 | Top5_MAE: 83.60 | Top10_MAE: 61.88 | Top1_MSE: 21723.33 | P95_MAE: 68.93

* **Fixed:**
  * Gradient Starvation: Fixed by completely removing Huber Loss (delta=0.1).
  * Dead ReLU & Oversmoothing: Fixed by adding `nn.LayerNorm(hidden_channels)` right before the ReLU activation. This retains hub information and normalizes inputs to keep ReLUs active.
  * No Stabilization / Weak Decoder: Fixed by introducing gradient clipping (GRAD_CLIP = 1.0) and the ReduceLROnPlateau scheduler. This stabilizes the gradients feeding into the final `GCNConv(hidden_channels, 1)` layer, turning a previously weak decoder into a functional one, taking the R² score from negative to positive.

---

### 1.2 GAT-IC AND LT
#### **Try 1:**
* **NN Structure:** 2-Layer GAT (Input --> GATConv [1 head] --> ELU --> Dropout --> GATConv [1 head] --> Squeeze). Hidden Dimension: 32, Dropout: 0.1.
* **Technique Used:** Graph Attention Network (GAT) with a single attention head for Node Regression (IC Influence Estimation).
* **Learning Rate (LR):** 1e-3 | **Scheduler:** None | **Optimizer:** AdamW (Weight Decay: 1e-4) | **Epochs:** 1000 (Early Stopping Patience: 100). Hit max epoch 1000.
* **GAT IC EXPERIMENT RESULTS:**
  * Best Epoch: 1000 | Best Validation MAE: 21.451452
  * Test Metrics: MAE: 20.869232 | RMSE: 27.877210 | R²: 0.106440 | Spearman: 0.556013 | NDCG@10: 0.579796 | Precision@10: 0.200000
* **GAT LT EXPERIMENT RESULTS:**
  * Best Epoch: 707 | Best Validation MAE: 6.875693
  * Test Metrics: MAE: 6.244497 | RMSE: 10.358725 | R²: -0.013859 | Spearman: 0.455578 | NDCG@10: 0.227910 | Precision@10: 0.100000
* **Problems:**
  * Gradient Starvation: Still utilizing HuberLoss(delta=0.1), which restricts gradients for large target scales.
  * Weak Decoder & Lack of Stabilization: Missing learning rate scheduling, gradient clipping, and intermediate normalization (like LayerNorm), leaving the output projection bottlenecked.
  * Stagnant Training: The model hit the hard ceiling of 1000 epochs, indicating the optimizer struggled to converge effectively without a scheduler.

#### **Try 2:**
* **NN Structure:** 2-Layer GAT with Stabilization (Input → GATConv [1 head] → LayerNorm → ELU → Dropout → GATConv [1 head] / Decoder Head → Squeeze). Hidden: 32, Dropout: 0.10.
* **Technique Used:** GAT for IC Influence Estimation with Masked Loss.
* **LR:** 1e-3 (Minimum LR: 1e-5) | **Scheduler:** ReduceLROnPlateau (Factor: 0.5, Patience: 20, Mode: Min).
* **Optimizer:** AdamW (Weight Decay: 1e-4, Gradient Clip: 1.0). Loss: MSELoss.
* **Epochs:** 500 (Early Stopping Patience: 100).

**GAT IC EXPERIMENT RESULTS:**
* Best Epoch: 108 | Best Validation MAE: 6.471844
* Test Metrics: MAE: 7.9434 | RMSE: 14.8504 | R²: -0.0138 | Spearman: 0.3734 | KendallTau: 0.2563
* Ranking: Precision@10: 0.2 | Precision@20: 0.25 | Recall@10: 0.2 | Recall@20: 0.25 | NDCG@10: 0.3073 | NDCG@20: 0.4173 | Top10Overlap: 2 | Top20Overlap: 5
* Elite Diagnostics: Top1_MAE: 105.08 | Top5_MAE: 63.20 | Top10_MAE: 45.51 | Top1_MSE: 11042.37 | P95_MAE: 51.04

**GAT LT EXPERIMENT RESULTS:**
* Best Epoch: 145 | Best Validation MAE: 10.072399
* Test Metrics: MAE: 11.3070 | RMSE: 21.7938 | R²: -0.0217 | Spearman: 0.3655 | KendallTau: 0.2449
* Ranking: Precision@10: 0.3 | Precision@20: 0.35 | Recall@10: 0.3 | Recall@20: 0.35 | NDCG@10: 0.2955 | NDCG@20: 0.4005 | Top10Overlap: 3 | Top20Overlap: 7
* Elite Diagnostics: Top1_MAE: 165.55 | Top5_MAE: 92.20 | Top10_MAE: 68.01 | Top1_MSE: 27407.83 | P95_MAE: 75.98

* **Problem solved:**
  * Loss & Optimization Issues: Swapped to MSELoss and added the ReduceLROnPlateau scheduler alongside Gradient Clipping (GRAD_CLIP = 1.0) to prevent gradient starvation and stabilize learning.
  * Dead Neurons & Instability: Introduced LayerNorm and swapped to the ELU activation function to keep gradients flowing smoothly.
  * Feature Loss & Weak Output: Added a Residual Connection to retain initial node information and strengthened the output with a dedicated Decoder Head.
  * Lost Training Progress: Implemented Checkpointing to ensure the best model state (lowest Validation MAE) is saved and reloaded for testing, preventing the model from just running out the clock.

---

### 1.3 GIN-IC-LT
#### **Try 1:**
* **Structure:** 2-Layer GIN. Layer 1 MLP: Linear --> ReLU --> Linear. Intermediate: ReLU --> Dropout (0.1). Layer 2 MLP: Linear --> ReLU --> Linear (Output 1) --> Squeeze. Hidden Dimension: 32.
* **Technique Used:** Graph Isomorphism Network (GIN) for Node Regression (IC Influence Estimation).
* **LR (Learning Rate):** 1e-3 | **Scheduler:** None | **Optimizer:** AdamW (Weight Decay: 1e-4) | **Loss:** Huber (delta=0.1) | **Epochs:** 1000 (Early Stopping Patience: 100).
* **GIN IC EXPERIMENT RESULTS:**
  * Best Epoch: 967 | Best Validation MAE: 20.531956
  * Test Metrics: MAE: 20.838774 | RMSE: 30.287656 | R²: -0.054767 | Spearman: 0.557053 | NDCG@10: 0.729419 | Precision@10: 0.300000
* **GIN LT EXPERIMENT RESULTS:**
  * Best Epoch: 999 | Best Validation MAE: 6.123455
  * Test Metrics: MAE: 4.888311 | RMSE: 8.505208 | R²: 0.316506 | Spearman: 0.722711 | NDCG@10: 0.588441 | Precision@10: 0.200000
* **Problems:**
  * Gradient Starvation: Still using HuberLoss(delta=0.1), heavily restricting gradients for your larger IC influence targets.
  * Dead Neurons & Instability: Missing LayerNorm to stabilize the MLPs inside the GINConvs.
  * Weak Optimization: Lacks a learning rate scheduler and gradient clipping to smoothly guide the model to convergence.

#### **Try 2:**
* **NN Structure:** 2-Layer GIN with Stabilization.
  * Layer 1: GINConv (MLP: Linear --> ReLU --> Linear) --> LayerNorm --> ReLU --> Dropout (0.1).
  * Layer 2: GINConv (MLP: Linear --> ReLU --> Linear [Output 1]) --> Squeeze.
  * Hidden Dimension: 32.
* **Technique Used:** Graph Isomorphism Network (GIN) for Node Regression (IC Influence Estimation).
* **LR:** 1e-3 (Minimum LR: 1e-5) | **Scheduler:** ReduceLROnPlateau (Factor: 0.5, Patience: 20, Mode: Min).
* **Optimizer:** AdamW (Weight Decay: 1e-4, Gradient Clip: 1.0) | **Loss:** MSELoss | **Epochs:** 500 (Early Stopping Patience: 100). Includes Checkpointing to restore the best validation state.

**GIN IC EXPERIMENT SUMMARY:**
* Best Epoch: 498 | Best Validation MAE: 6.7702
* Test Performance: MAE: 7.1003 | RMSE: 11.4153 | R²: 0.4009 | Spearman: 0.4746 | KendallTau: 0.3291
* Ranking: Precision@10: 0.6000 | Precision@20: 0.4500 | Recall@10: 0.6000 | Recall@20: 0.4500 | NDCG@10: 0.7742 | NDCG@20: 0.7408 | Top10Overlap: 6.0 | Top20Overlap: 9.0
* Elite Diagnostics: Top1_MAE: 58.6293 | Top5_MAE: 42.4026 | Top10_MAE: 29.1024 | Top1_MSE: 3437.3901 | P95_MAE: 32.3400

**GIN LT EXPERIMENT SUMMARY:**
* Best Epoch: 393 | Best Validation MAE: 10.3982
* Test Performance: MAE: 9.7537 | RMSE: 15.7019 | R²: 0.4696 | Spearman: 0.4572 | KendallTau: 0.3207
* Ranking: Precision@10: 0.6000 | Precision@20: 0.5000 | Recall@10: 0.6000 | Recall@20: 0.5000 | NDCG@10: 0.8553 | NDCG@20: 0.8089 | Top10Overlap: 6.0 | Top20Overlap: 10.0
* Elite Diagnostics: Top1_MAE: 86.2934 | Top5_MAE: 53.1905 | Top10_MAE: 42.3124 | Top1_MSE: 7446.5503 | P95_MAE: 46.9308

---

### 1.4 GRAPHSAGE-IC-LT
#### **Try 1:**
* **NN Structure:** 2-Layer GraphSAGE. Input → SAGEConv → ReLU → Dropout (0.1) → SAGEConv (Output mapped directly to 1) → Squeeze. Hidden Dimension: 32.
* **Technique Used:** GraphSAGE for Node Regression (IC Influence Estimation).
* **Learning Rate (LR):** 1e-3 | **Scheduler:** None used.
* **Optimizer:** AdamW (Weight Decay: 1e-4) | **Loss:** Huber (delta=0.1) | **Epochs:** 1000 (Early Stopping Patience: 100).
* **GRAPHSAGE IC EXPERIMENT RESULTS:**
  * Best Epoch: 1000 | Best Validation MAE: 21.044176
  * Test Metrics: MAE: 19.463604 | RMSE: 26.597049 | R²: 0.186623 | Spearman: 0.590983 | NDCG@10: 0.648676 | Precision@10: 0.300000
* **GRAPHSAGE LT EXPERIMENT RESULTS:**
  * Best Epoch: 1000 | Best Validation MAE: 6.559759
  * Test Metrics: MAE: 5.899995 | RMSE: 9.478703 | R²: 0.151088 | Spearman: 0.553276 | NDCG@10: 0.554863 | Precision@10: 0.400000
* **Problems:** Same problems here (Gradient starvation, weak optimization, missing stabilization).

#### **Try 2:**
* **NN Structure:** 2-Layer GraphSAGE with Stabilization.
  * Layer 1: SAGEConv (Input -> Hidden) -> LayerNorm -> ReLU -> Dropout (0.1).
  * Layer 2: SAGEConv (Hidden -> Output 1) -> Squeeze.
  * Hidden Dimension: 32.
* **Technique Used:** GraphSAGE for Node Regression (IC Influence Estimation).
* **Learning Rate (LR):** 1e-3 (Minimum LR: 1e-5) | **Scheduler:** ReduceLROnPlateau (Factor: 0.5, Patience: 20, Mode: Min).
* **Optimizer:** AdamW (Weight Decay: 1e-4, Gradient Clip: 1.0) | **Loss:** MSELoss | **Epochs:** 500 (Early Stopping Patience: 100).

**VANILLA GRAPHSAGE IC RESULTS:**
* Best Epoch: 500 | Best Validation MAE: 4.177852
* Regression Metrics: MAE: 5.387420 | RMSE: 10.636760 | R²: 0.479852 | Spearman: 0.777553 | KendallTau: 0.584369
* Ranking Metrics: Precision@10: 0.600000 | Precision@20: 0.700000 | Recall@10: 0.600000 | Recall@20: 0.700000 | NDCG@10: 0.793159 | NDCG@20: 0.830391 | Top10Overlap: 6.000000 | Top20Overlap: 14.000000
* Elite Diagnostics: Top1_MAE: 84.974396 | Top5_MAE: 43.680428 | Top10_MAE: 29.729282 | Top1_MSE: 7220.647949 | P95_MAE: 32.457703

**VANILLA GRAPHSAGE LT RESULTS:**
* Best Epoch: 500 | Best Validation MAE: 6.707730
* Regression Metrics: MAE: 8.337682 | RMSE: 16.702076 | R²: 0.399913 | Spearman: 0.758523 | KendallTau: 0.560223
* Ranking Metrics: Precision@10: 0.700000 | Precision@20: 0.700000 | Recall@10: 0.700000 | Recall@20: 0.700000 | NDCG@10: 0.756873 | NDCG@20: 0.782895 | Top10Overlap: 7.000000 | Top20Overlap: 14.000000
* Elite Diagnostics: Top1_MAE: 141.302597 | Top5_MAE: 70.318321 | Top10_MAE: 47.856438 | Top1_MSE: 19966.423828 | P95_MAE: 56.393631

---

### 1.5 PGAT
#### **Try 1:**
* **NN Structure:**
  * Input Projection: Linear -> GELU -> LayerNorm -> Dropout.
  * Encoder: 2-Layer PGATConv (4 Heads, Probability Gate, Double Probability Propagation, Residual Connections, LayerNorm).
  * Output Projection: Concatenated features -> Linear -> GELU -> BatchNorm1d -> Dropout.
  * Decoder: Dedicated IC Head (Linear -> GELU -> Linear).
* **Technique Used:** Probability-Guided Graph Attention Network (PGAT-v1) for Single-Task Node Regression (IC Influence Estimation only).
* **Learning Rate (LR):** 1e-3 (Utilizes Layer-wise LR Decay LLRD = 0.8) | **Scheduler:** Cosine Scheduler with Linear Warmup (Warmup: 20 epochs).
* **Optimizer:** AdamW (Weight Decay: 1e-4) | **Loss:** HuberLoss(delta=0.1) | **Epochs:** 1000 (Early Stopping Patience: 100).
* **PGAT IC RESULTS:**
  * Best Epoch: 748 | Best Validation MAE: 9.282394
  * Test Metrics: MAE: 9.991121 | RMSE: 12.893615 | R²: 0.808850 | Spearman: 0.884333 | NDCG@10: 0.924171 | Precision@10: 0.700000
* **PGAT-v1 LT FINAL RESULTS:**
  * Best Epoch: 214 | Best Validation MAE: 5.8727
  * LT Test Metrics: MAE: 5.5844 | RMSE: 8.9629 | R²: 0.2410 | Spearman: 0.6132 | NDCG@10: 0.4355 | Precision@10: 0.2000

#### **Try 2:**
* **NN Structure:** * Input Projection: Linear --> GELU -->LayerNorm --> Dropout.
  * PGAT Layers: 2 Layers of PGATConv (4 Heads). Features: Separate Source/Dest weights, Probability Gate, Degree Temperature Softmax, Double Probability Scaling, Residual Updates, and LayerNorm.
  * Output Projection: Concatenates raw features with learned embeddings --> Linear --> GELU --> BatchNorm1d --> Dropout --> Linear.
  * Decoder (IC Head & LT Head): Linear --> GELU --> Linear (Output 1) --> Squeeze.
  * Dimensions: Hidden: 32, Embed: 32, Dropout: 0.1.
* **Technique Used:** Probability-Guided Graph Attention Network (PGAT) for Node Regression (IC Influence Estimation).
* **Learning Rate (LR):** 1e-3 (Minimum LR: 1e-5) | **Scheduler:** ReduceLROnPlateau (Factor: 0.5, Patience: 20, Mode: Min).
* **Optimizer:** AdamW (Weight Decay: 1e-4, Gradient Clip: 1.0) | **Loss:** MSELoss | **Epochs:** 500 (Early Stopping Patience: 100).

**VANILLA PGAT IC RESULTS:**
* Best Epoch: 74 | Best Validation MAE: 4.113091
* Regression Metrics: MAE: 5.625539 | RMSE: 10.427214 | R²: 0.500145 | Spearman: 0.700809 | KendallTau: 0.518112
* Ranking Metrics: Precision@10: 0.600000 | Precision@20: 0.750000 | Recall@10: 0.600000 | Recall@20: 0.750000 | NDCG@10: 0.763238 | NDCG@20: 0.783458 | Top10Overlap: 6.000000 | Top20Overlap: 15.000000
* Elite Diagnostics: Top1_MAE: 77.974586 | Top5_MAE: 42.178062 | Top10_MAE: 28.369192 | Top1_MSE: 6080.036133 | P95_MAE: 30.833004

**VANILLA PGAT LT RESULTS:**
* Best Epoch: 178 | Best Validation MAE: 5.907985
* Regression Metrics: MAE: 8.773910 | RMSE: 16.047848 | R²: 0.446004 | Spearman: 0.652626 | KendallTau: 0.476381
* Ranking Metrics: Precision@10: 0.700000 | Precision@20: 0.600000 | Recall@10: 0.700000 | Recall@20: 0.600000 | NDCG@10: 0.692531 | NDCG@20: 0.725631 | Top10Overlap: 7.000000 | Top20Overlap: 12.000000
* Elite Diagnostics: Top1_MAE: 129.969757 | Top5_MAE: 62.425499 | Top10_MAE: 41.045704 | Top1_MSE: 16892.138672 | P95_MAE: 49.045399
