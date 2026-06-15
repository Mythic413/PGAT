import gzip

print("Inspecting retweet network...\n")

with gzip.open("data/higgs-retweet_network.edgelist.gz", "rt") as f:
    for i in range(20):
        print(f.readline().strip())