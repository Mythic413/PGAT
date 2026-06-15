import gzip

print("Inspecting activity file...\n")

with gzip.open("data/higgs-activity_time.txt.gz", "rt") as f:
    for i in range(20):
        line = f.readline().strip()
        print(line)