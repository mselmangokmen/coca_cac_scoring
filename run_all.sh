#!/usr/bin/env bash
# The whole pipeline, both encoders, both COCA arms. Edit the three paths, then: bash run_all.sh
#   DATA     the COCA release root (holding Gated_release_final/ and deidentified_nongated/)
#   CURIA    hf download raidium/curia config.json model.safetensors --local-dir weights/curia
#   MEDDINO  hf download ricklisz123/MedDINOv3-ViTB-16-CT-3M model.pth --local-dir weights/meddinov3
#   DINOV3   git clone https://github.com/facebookresearch/dinov3 dinov3
set -euo pipefail
DATA=${DATA:-/data/coca}
CURIA=${CURIA:-weights/curia}
MEDDINO=${MEDDINO:-weights/meddinov3/model.pth}
DINOV3=${DINOV3:-dinov3}

# 1. reference scores (gated: recomputed from the contours; non-gated: Stanford's scores.xlsx) and the
#    patient-level splits. All four CSVs are shipped in data/ and splits/; this regenerates them.
python scripts/build_gated_reference.py --data-root "$DATA" --out data/coca_gated_reference.csv
python scripts/export_nongated_reference.py --data-root "$DATA" --out data/coca_nongated_reference.csv
python scripts/make_splits.py --gated-reference data/coca_gated_reference.csv \
    --nongated-reference data/coca_nongated_reference.csv --out-dir splits

# 2. frozen features
for C in gated nongated; do
  python scripts/extract_features.py --encoder curia --weights "$CURIA" --cohort $C --data-root "$DATA" \
      --split-csv splits/coca_${C}_split.csv
  python scripts/extract_features.py --encoder meddinov3 --weights "$MEDDINO" --dinov3-repo "$DINOV3" --cohort $C \
      --data-root "$DATA" --split-csv splits/coca_${C}_split.csv
done

# 3. scoring heads: train on the train side, evaluate on the test side, per arm and encoder
for E in curia_L7 meddinov3_L11; do
  for C in gated nongated; do
    python scripts/train_head.py --features outputs/features/$E/coca_$C.npz --split-csv splits/coca_${C}_split.csv \
        --out outputs/heads/$E/coca_$C
  done
done

# 4. localization on the gated test side, every layer
python scripts/pointing.py --data-root "$DATA" --split-csv splits/coca_gated_split.csv --side test \
    --encoder curia="$CURIA" --encoder meddinov3="$MEDDINO" --dinov3-repo "$DINOV3"

# 5. layer comparison: the same features and heads from the last block (L12)
for C in gated nongated; do
  python scripts/extract_features.py --encoder curia --weights "$CURIA" --layer 12 --cohort $C --data-root "$DATA" \
      --split-csv splits/coca_${C}_split.csv
  python scripts/extract_features.py --encoder meddinov3 --weights "$MEDDINO" --dinov3-repo "$DINOV3" --layer 12 \
      --cohort $C --data-root "$DATA" --split-csv splits/coca_${C}_split.csv
done
for E in curia_L12 meddinov3_L12; do
  for C in gated nongated; do
    python scripts/train_head.py --features outputs/features/$E/coca_$C.npz --split-csv splits/coca_${C}_split.csv \
        --out outputs/heads/$E/coca_$C
  done
done
