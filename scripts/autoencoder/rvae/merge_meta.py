import glob
import json
import os

path_root = os.path.join(os.environ["SCRATCH_ROOT"], "caches/autoencoder_cache_scenario_control")
output_file_name = "sequence_token_mapping.json"
path_json_output = os.path.join(path_root, output_file_name)

# Discover all per-map / per-split mappings: start with "sequence_token_mapping"
# and end with either "trainval.json" or "test.json"
input_file_names = sorted(
    os.path.basename(p)
    for suffix in ("trainval.json", "test.json")
    for p in glob.glob(os.path.join(path_root, f"sequence_token_mapping*{suffix}"))
)

if not input_file_names:
    raise FileNotFoundError(f"No sequence_token_mapping *trainval/*test json files found in {path_root}")

# Merge all mappings (later files overwrite earlier ones on key collision)
merged = {}
for file_name in input_file_names:
    path_json = os.path.join(path_root, file_name)
    if not os.path.exists(path_json):
        print(f"WARNING: {path_json} not found, skipping.")
        continue

    with open(path_json, "r") as f:
        data = json.load(f)

    overlap = merged.keys() & data.keys()
    if overlap:
        print(f"WARNING: {len(overlap)} overlapping keys in {file_name} will be overwritten.")

    merged.update(data)
    print(f"Loaded {len(data)} sequences from {file_name} (total: {len(merged)}).")

# Save merged JSON
with open(path_json_output, "w") as f:
    json.dump(merged, f, indent=4)

print(f"Merged {len(merged)} sequences into {path_json_output}")
