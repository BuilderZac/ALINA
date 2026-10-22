#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"
OUTPUT_FILE="${OUTPUT_FILE:-outputs/readme_baseline_evaluation.txt}"

CANNY_DIRS=(
    data/gt_alina_labels/canny_textfiles/canny_textfiles_1
    data/gt_alina_labels/canny_textfiles/canny_textfiles_2
    data/gt_alina_labels/canny_textfiles/canny_textfiles_3
)

ALINA_DIRS=(
    data/gt_alina_labels/ALINA_textfiles/ALINA_textfiles_1
    data/gt_alina_labels/ALINA_textfiles/ALINA_textfiles_2
    data/gt_alina_labels/ALINA_textfiles/ALINA_textfiles_3
)

for directory in "${CANNY_DIRS[@]}" "${ALINA_DIRS[@]}"; do
    if [[ ! -d "${directory}" ]]; then
        printf 'Missing evaluation directory: %s\n' "${directory}" >&2
        exit 1
    fi
done

mkdir -p "$(dirname -- "${OUTPUT_FILE}")"

"${PYTHON}" cli.py evaluate \
    --canny-dirs \
        "${CANNY_DIRS[0]}" \
        "${CANNY_DIRS[1]}" \
        "${CANNY_DIRS[2]}" \
    --alina-dirs \
        "${ALINA_DIRS[0]}" \
        "${ALINA_DIRS[1]}" \
        "${ALINA_DIRS[2]}" \
    | tee "${OUTPUT_FILE}"

printf '\nSaved evaluation to: %s\n' "${OUTPUT_FILE}"

