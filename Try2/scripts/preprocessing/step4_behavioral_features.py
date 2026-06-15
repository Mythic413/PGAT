import gzip
import pandas as pd
from collections import defaultdict
import networkx as nx
from pathlib import Path

print("Loading Higgs-5000 nodes...")

TRY2_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = TRY2_DIR.parent

GRAPH_PATH = TRY2_DIR / "data" / "higgs_5000.edgelist"

ACTIVITY_PATH = (
    PROJECT_DIR
    / "data"
    / "higgs-activity_time.txt.gz"
)

OUTPUT_PATH = (
    TRY2_DIR
    / "data"
    / "behavioral_features.csv"
)
# ==================================================
# Load Higgs-5000 Graph
# ==================================================

G = nx.read_edgelist(
    GRAPH_PATH,
    nodetype=int,
    create_using=nx.DiGraph()
)

nodes = sorted(G.nodes())
node_set = set(nodes)

print(f"Nodes in subgraph: {len(nodes):,}")

# ==================================================
# Statistics Containers
# ==================================================

activity_count = defaultdict(int)

rt_count = defaultdict(int)
mt_count = defaultdict(int)
re_count = defaultdict(int)

first_time = {}
last_time = {}

print("\nProcessing activity file...")

processed = 0

# ==================================================
# Parse Activity File
# ==================================================

with gzip.open(ACTIVITY_PATH, "rt") as f:

    for line in f:

        user1, user2, timestamp, action = (
            line.strip().split()
        )

        user1 = int(user1)
        timestamp = int(timestamp)

        # Only consider users in Higgs-5000
        if user1 not in node_set:
            continue

        processed += 1

        activity_count[user1] += 1

        # ------------------------------------------
        # Action Counts
        # ------------------------------------------

        if action == "RT":
            rt_count[user1] += 1

        elif action == "MT":
            mt_count[user1] += 1

        elif action == "RE":
            re_count[user1] += 1

        # ------------------------------------------
        # Robust Timestamp Tracking
        # ------------------------------------------

        if user1 not in first_time:

            first_time[user1] = timestamp
            last_time[user1] = timestamp

        else:

            first_time[user1] = min(
                first_time[user1],
                timestamp
            )

            last_time[user1] = max(
                last_time[user1],
                timestamp
            )

print(f"Processed relevant activities: {processed:,}")

# ==================================================
# Build Behavioral Feature Matrix
# ==================================================

print("\nBuilding Raw Behavioral Feature Matrix...")

rows = []

for node in nodes:

    total = activity_count[node]

    # ------------------------------------------
    # Action Ratios
    # ------------------------------------------

    if total > 0:

        rt_ratio = rt_count[node] / total
        mt_ratio = mt_count[node] / total
        re_ratio = re_count[node] / total

    else:

        rt_ratio = 0.0
        mt_ratio = 0.0
        re_ratio = 0.0

    # ------------------------------------------
    # Activity Span
    # ------------------------------------------

    span = (
        last_time.get(node, 0)
        - first_time.get(node, 0)
    )
    
    is_active = int(total > 0)

    rows.append([
        node,
        total,
        rt_ratio,
        mt_ratio,
        re_ratio,
        span,
        is_active
    ])

behavior = pd.DataFrame(
    rows,
    columns=[
        "node",
        "activity_count",
        "retweet_ratio",
        "mention_ratio",
        "reply_ratio",
        "activity_span",
        "is_active"
    ]
)

# ==================================================
# Raw Statistics
# ==================================================

print("\nRaw Behavioral Statistics:")

print(
    behavior[
        [
            "activity_count",
            "retweet_ratio",
            "mention_ratio",
            "reply_ratio",
            "activity_span",
            "is_active"
        ]
    ].describe()
)

# ==================================================
# Save RAW Features
# ==================================================

behavior.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\nBehavior Matrix Shape:")
print(behavior.shape)

print("\nFeature Columns:")
print(behavior.columns.tolist())

print("\nStep 4 completed successfully. Unnormalized features ")
