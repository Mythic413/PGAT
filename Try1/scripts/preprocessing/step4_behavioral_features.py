import gzip
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import networkx as nx

print("Loading Higgs-5000 nodes...")

G = nx.read_edgelist(
    "data/higgs_5000.edgelist",
    nodetype=int,
    create_using=nx.DiGraph()
)

nodes = sorted(G.nodes())
node_set = set(nodes)

print(f"Nodes in subgraph: {len(nodes)}")

# Statistics containers
activity_count = defaultdict(int)

rt_count = defaultdict(int)
mt_count = defaultdict(int)
re_count = defaultdict(int)

first_time = {}
last_time = {}

print("\nProcessing activity file...")

processed = 0

with gzip.open("data/higgs-activity_time.txt.gz", "rt") as f:
    for line in f:
        user1, user2, timestamp, action = line.strip().split()

        user1 = int(user1)
        timestamp = int(timestamp)

        # Only consider users in Higgs-5000
        if user1 not in node_set:
            continue

        processed += 1

        activity_count[user1] += 1

        if action == "RT":
            rt_count[user1] += 1
        elif action == "MT":
            mt_count[user1] += 1
        elif action == "RE":
            re_count[user1] += 1

        if user1 not in first_time:
            first_time[user1] = timestamp

        last_time[user1] = timestamp

print(f"Processed relevant activities: {processed:,}")

print("\nBuilding behavioral feature matrix...")

rows = []

for node in nodes:

    total = activity_count[node]

    if total > 0:
        rt_ratio = rt_count[node] / total
        mt_ratio = mt_count[node] / total
        re_ratio = re_count[node] / total
    else:
        rt_ratio = 0.0
        mt_ratio = 0.0
        re_ratio = 0.0

    span = (
        last_time.get(node, 0)
        - first_time.get(node, 0)
    )

    rows.append([
        node,
        total,
        rt_ratio,
        mt_ratio,
        re_ratio,
        span
    ])

behavior = pd.DataFrame(
    rows,
    columns=[
        "node",
        "activity_count",
        "retweet_ratio",
        "mention_ratio",
        "reply_ratio",
        "activity_span"
    ]
)

print("\nRaw Behavioral Statistics:")
print(behavior.describe())

feature_cols = [
    "activity_count",
    "retweet_ratio",
    "mention_ratio",
    "reply_ratio",
    "activity_span"
]

scaler = MinMaxScaler()

behavior[feature_cols] = scaler.fit_transform(
    behavior[feature_cols]
)

print("\nNormalized Behavioral Statistics:")
print(behavior[feature_cols].describe())

behavior.to_csv(
    "data/behavioral_features.csv",
    index=False
)

print("\nSaved:")
print("data/behavioral_features.csv")

print("\nBehavior Matrix Shape:")
print(behavior.shape)