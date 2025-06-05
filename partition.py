import os
import pickle
import re

# === Input directories ===
passage_dir = "data/indices/passages/s2orc"
embedding_dir = "data/indices/embeddings/facebook/contriever-msmarco/s2orc"

# === Output directories ===
out_dirs = {
    "top": {
        "passages": "data-1/passages",
        "embeddings": "data-1/embeddings"
    },
    "bottom": {
        "passages": "data-2/passages",
        "embeddings": "data-2/embeddings"
    }
}

# === Ensure output dirs exist ===
for split in out_dirs.values():
    os.makedirs(split["passages"], exist_ok=True)
    os.makedirs(split["embeddings"], exist_ok=True)

# === Filename normalization ===
def normalize_passage_name(pkl_name):
    # Convert: raw_passages_1-26-of-28.pkl -> passages1_26.pkl
    match = re.match(r"raw_passages_(\d+)-(\d+)-of-\d+\.pkl", pkl_name)
    if not match:
        raise ValueError(f"Unexpected passage filename format: {pkl_name}")
    x, y = match.groups()
    return f"passages{x}_{y}.pkl"

# === Load and verify ===
all_passages = []
all_embeddings = []

for pkl_file in sorted(os.listdir(passage_dir)):
    if not pkl_file.endswith(".pkl") or not pkl_file.startswith("raw_passages_"):
        continue

    try:
        embedding_file = normalize_passage_name(pkl_file)
    except ValueError as e:
        print(e)
        continue

    passage_path = os.path.join(passage_dir, pkl_file)
    embedding_path = os.path.join(embedding_dir, embedding_file)

    if not os.path.exists(embedding_path):
        print(f"Skipping {pkl_file}: no embedding file found at {embedding_file}")
        continue

    with open(passage_path, "rb") as f:
        passages = pickle.load(f)
    with open(embedding_path, "rb") as f:
        embeddings = pickle.load(f)

    if len(passages) != len(embeddings):
        print(f"Length mismatch: {pkl_file} ({len(passages)} vs {len(embeddings)})")
        continue

    all_passages.extend(passages)
    all_embeddings.extend(embeddings)

# === Split ===
total = len(all_passages)
half = total // 2

top_passages = all_passages[:half]
top_embeddings = all_embeddings[:half]
bottom_passages = all_passages[half:]
bottom_embeddings = all_embeddings[half:]

# === Save ===
with open(os.path.join(out_dirs["top"]["passages"], "part.pkl"), "wb") as f:
    pickle.dump(top_passages, f)

with open(os.path.join(out_dirs["top"]["embeddings"], "part.pkl"), "wb") as f:
    pickle.dump(top_embeddings, f)

with open(os.path.join(out_dirs["bottom"]["passages"], "part.pkl"), "wb") as f:
    pickle.dump(bottom_passages, f)

with open(os.path.join(out_dirs["bottom"]["embeddings"], "part.pkl"), "wb") as f:
    pickle.dump(bottom_embeddings, f)

print(f"✅ Done. Total: {total}, Top: {half}, Bottom: {total - half}")
