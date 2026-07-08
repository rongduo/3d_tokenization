#!/usr/bin/env bash
# Prepare parts_partsam.pt for 3DCoMPaT200 eval (test) objects.
#
# Pipeline (matches training: GLTF symlinks, no GLB conversion):
#   1) list all eval object dirs -> test_obj_list.txt
#   2) symlink source .gltf files for PartSAM ValDataset
#   3) PartSAM GPU inference (partsam env + GPU)
#   4) face vote + graph cut -> parts_partsam.pt
#
# Usage:
#   bash pipes_tools/run_partsam_eval_prep.sh list
#   bash pipes_tools/run_partsam_eval_prep.sh link
#   bash pipes_tools/run_partsam_eval_prep.sh infer
#   bash pipes_tools/run_partsam_eval_prep.sh convert

set -u

PYTHON_BIN="${PYTHON_BIN:-/home/jl/anaconda3/envs/newpipelinefind3d/bin/python}"
PARTSAM_PYTHON="${PARTSAM_PYTHON:-/home/jl/anaconda3/envs/PartSAM/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARTSAM_ROOT="${PARTSAM_ROOT:-/data5/jl/project/PartSAM}"

EVAL_ROOT="${EVAL_ROOT:-/data5/jl/project/tokenizer_seg/cosmo3d_dataset__d3compat_and_partspt/3dcompat200}"
MESH_ROOT="${MESH_ROOT:-/data3/jl/dataset/3DCoMPaT200/datasets--CoMPaT--3DCoMPaT200/snapshots/a19e536383845527203a1f3bf0b509e12ed42bd7/Compat200/models}"

WORK_DIR="${WORK_DIR:-$PROJECT_ROOT/pipes_tools/eval_partsam_work}"
OBJ_LIST="${OBJ_LIST:-$WORK_DIR/test_obj_list.txt}"
GLTF_DIR="${GLTF_DIR:-$PARTSAM_ROOT/compat_eval/gltf_test_all}"
INFER_DIR="${INFER_DIR:-$PARTSAM_ROOT/infer_out/test_eval}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

step="${1:-help}"

run_list() {
    echo "=== Step 1: list eval object directories ==="
    mkdir -p "$WORK_DIR"
    "$PYTHON_BIN" "$PROJECT_ROOT/pipes_tools/list_d3compat200_objects.py" \
        --data_root "$EVAL_ROOT" \
        --out_txt "$OBJ_LIST" \
        --require_points
    echo "OBJ_LIST=$OBJ_LIST"
}

run_link() {
    echo "=== Step 2: symlink GLTF (no GLB conversion) ==="
    if [[ ! -f "$OBJ_LIST" ]]; then run_list; fi
    "$PYTHON_BIN" "$PROJECT_ROOT/pipes_tools/link_gltf_for_partsam.py" \
        --obj_list "$OBJ_LIST" \
        --mesh_root "$MESH_ROOT" \
        --out_dir "$GLTF_DIR"
    echo "GLTF_DIR=$GLTF_DIR"
}

run_infer() {
    echo "=== Step 3: PartSAM GPU inference (reads .gltf directly) ==="
    if [[ ! -d "$GLTF_DIR" ]] || [[ -z "$(ls -A "$GLTF_DIR"/*.gltf 2>/dev/null)" ]]; then
        echo "GLTF dir empty or missing: $GLTF_DIR (run: $0 link)" >&2
        exit 1
    fi
    mkdir -p "$INFER_DIR"
    cd "$PARTSAM_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" SAVE_DIR="$INFER_DIR" "$PARTSAM_PYTHON" \
        evaluation/infer_save_masks.py \
        "dataset.root_dir=$GLTF_DIR"
    echo "INFER_DIR=$INFER_DIR"
}

run_convert() {
    echo "=== Step 4: infer -> parts_partsam.pt (face vote + graph cut) ==="
    if [[ ! -f "$OBJ_LIST" ]]; then run_list; fi
    if [[ ! -d "$INFER_DIR" ]]; then
        echo "Infer dir missing: $INFER_DIR (run: $0 infer)" >&2
        exit 1
    fi
    export PARTSAM_ROOT
    "$PYTHON_BIN" "$PROJECT_ROOT/pipes_tools/convert_partsam_infer_to_eval.py" \
        --infer_dir "$INFER_DIR" \
        --obj_list "$OBJ_LIST" \
        --num_workers 32 \
        --iou_threshold 0.65 \
        --max_faces 50000 \
        --skip_existing
}

case "$step" in
    list) run_list ;;
    link|glb) run_link ;;   # glb kept as alias for backward compat
    infer) run_infer ;;
    convert) run_convert ;;
    all)
        run_list
        run_link
        echo ""
        echo "Next: PartSAM inference (GPU, partsam env):"
        echo "  bash $0 infer"
        echo "Then convert:"
        echo "  bash $0 convert"
        ;;
    help|*)
        cat <<EOF
Usage: bash $0 {list|link|infer|convert|all}

  list     - scan 3dcompat200 -> test_obj_list.txt
  link     - symlink source .gltf for PartSAM (no GLB conversion)
  infer    - PartSAM GPU inference (ValDataset reads .gltf, single GPU)
  convert  - graph cut + write parts_partsam.pt
  all      - list + link

Multi-GPU inference (recommended for 1692 meshes):
  GPUS=1,2,3,4,5,6,7 bash pipes_tools/run_partsam_eval_infer_multigpu.sh infer

Training used the same approach: compat_eval/gltf_gpu{0..6}/ symlinks -> source .gltf

Env overrides:
  EVAL_ROOT, MESH_ROOT, WORK_DIR, GLTF_DIR, INFER_DIR
  PYTHON_BIN, PARTSAM_PYTHON, CUDA_VISIBLE_DEVICES
EOF
        ;;
esac
