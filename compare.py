import json

file1 = "/home/ubuntu/Jinjian/retrieval-scaling/data/gpqa:0shot_cot::retrieval_q_retrieved_results.jsonl"
file2 = "/home/ubuntu/Jinjian/retrieval-scaling/faiss_ivfpq_search_531_IVFPQ.2048.256.256/gpqa:0shot_cot::retrieval_q_retrieved_results.jsonl"

def load_retrieval_texts(path):
    results = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            ctxs = entry.get("ctxs", [])
            # Extract only (id, retrieval_text) tuples
            ctx_entries = [(str(ctx.get("id")), ctx.get("retrieval text", "").strip()) for ctx in ctxs]
            results.append(ctx_entries)
    return results

def main():
    data1 = load_retrieval_texts(file1)
    data2 = load_retrieval_texts(file2)

    assert len(data1) == len(data2), "Files have different number of entries!"

    num_different = 0
    for i, (entry1, entry2) in enumerate(zip(data1, data2)):
        if entry1 != entry2:
            num_different += 1

    print(f"Total entries compared: {len(data1)}")
    print(f"Entries with different retrieval text values: {num_different}")

if __name__ == "__main__":
    main()
