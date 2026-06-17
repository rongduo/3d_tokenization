"""
Batch pre-compute label_embeds_siglip.pt for all training objects.
Reads mask_labels.txt from each object dir, encodes with SigLIP, normalizes, saves.
"""
import torch
import os
import sys
from transformers import AutoTokenizer, AutoModel

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
TRAIN_TXT = "/data5/jl/project/tokenizer_seg/cosmo3d_dataset__d3compat_and_partspt/d3compat/train.txt"
BATCH_SIZE = 64  # encode multiple label-sets at once for efficiency


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP on {device} ...")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model = AutoModel.from_pretrained(SIGLIP_MODEL_ID, local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_ID, local_files_only=True)
    model.eval()

    with open(TRAIN_TXT, "r") as f:
        obj_dirs = [line.strip() for line in f if line.strip()]

    total = len(obj_dirs)
    done = 0
    skipped = 0

    # Collect all (obj_dir, labels) first so we can batch-encode
    all_labels = []
    for d in obj_dirs:
        label_path = os.path.join(d, "mask_labels.txt")
        out_path = os.path.join(d, "label_embeds_siglip.pt")
        if os.path.isfile(out_path):
            skipped += 1
            all_labels.append(None)  # placeholder
            continue
        if not os.path.isfile(label_path):
            print(f"WARNING: missing mask_labels.txt in {d}")
            all_labels.append(None)
            continue
        with open(label_path, "r") as f:
            all_labels.append(f.read().splitlines())

    # Now batch-encode only the missing ones
    pending_labels = [(i, labels) for i, labels in enumerate(all_labels) if labels is not None]
    print(f"Total objects: {total}, already have embeds: {skipped}, need to generate: {len(pending_labels)}")

    for batch_start in range(0, len(pending_labels), BATCH_SIZE):
        batch = pending_labels[batch_start:batch_start + BATCH_SIZE]
        # tokenize: each label-set may have different number of parts, encode each label individually
        all_flat_labels = []
        group_sizes = []
        for _, labels in batch:
            all_flat_labels.extend(labels)
            group_sizes.append(len(labels))

        inputs = tokenizer(
            all_flat_labels,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            feats = model.get_text_features(**inputs)  # [total_labels, 768]
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
        feats = feats.cpu()

        # Split back per object and save
        offset = 0
        for idx, (orig_i, _) in enumerate(batch):
            n = group_sizes[idx]
            obj_feats = feats[offset:offset + n]
            offset += n
            out_path = os.path.join(obj_dirs[orig_i], "label_embeds_siglip.pt")
            torch.save(obj_feats, out_path)

        done += len(batch)
        if done % 500 == 0 or done >= len(pending_labels):
            print(f"Progress: {done}/{len(pending_labels)}")

    print(f"Done. Generated: {done}, skipped (already existed): {skipped}")


if __name__ == "__main__":
    main()
