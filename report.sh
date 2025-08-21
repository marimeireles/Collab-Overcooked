#!/usr/bin/env bash
# Usage: ./report.sh /nas/ucb/marimeireles/dev/Collab-Overcooked/final-data > completeness.md
set -euo pipefail

ROOT="${1:-/nas/ucb/marimeireles/dev/Collab-Overcooked/}"

# Expected recipe directories (model subdirs do NOT include the numeric prefixes)
expected_with_nums=(
  "1_baked_bell_pepper"
  "1_baked_sweet_potato"
  "1_boiled_egg"
  "1_boiled_mushroom"
  "1_boiled_sweet_potato"
  "2_boiled_potato_slices"
  "2_boiled_green_bean_slices"
  "2_boiled_corn_slices"
  "2_baked_pumpkin_slices"
  "3_baked_bell_pepper_soup"
  "3_baked_carrot_soup"
  "3_baked_mushroom_soup"
  "3_baked_potato_soup"
  "3_baked_pumpkin_soup"
  "5_mashed_broccoli_and_bean_patty"
  "5_mashed_carrot_and_chickpea_patty"
  "5_mashed_cauliflower_and_lentil_patty"
  "5_mashed_potato_and_pea_patty"
  "5_mashed_sweet_potato_and_bean_patty"
)

# Generate Markdown
echo "# Completeness report"
echo
echo "- Root: \`$ROOT\`"
echo "- Each task should contain **5** JSON files."
echo

# Iterate model-combination directories (depth 1)
while IFS= read -r model_dir; do
  model_name="$(basename "$model_dir")"
  echo "## $model_name"
  any_missing=0

  for item in "${expected_with_nums[@]}"; do
    pretty="$item"
    subdir="${item#*_}"             # strip numeric prefix, e.g., "1_baked..." -> "baked..."
    path="$model_dir/$subdir"

    count=0
    if [[ -d "$path" ]]; then
      # count JSON files directly inside subdir
      count=$(find "$path" -maxdepth 1 -type f -name '*.json' | wc -l | awk '{print $1}')
    fi
    missing=$((10 - count))
    status="($count/10"
    if (( missing > 0 )); then
      status="$status, missing $missing)"
      any_missing=1
    else
      status="$status, complete)"
    fi

    echo "- $pretty — $status"
  done

  if (( any_missing == 0 )); then
    echo "- All expected tasks complete."
  fi
  echo
done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | grep -E '_.*_' | sort)

