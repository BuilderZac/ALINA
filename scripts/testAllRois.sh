#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"
MODE="${1:-}"
RUN_LABELING="${RUN_LABELING:-1}"
OUTPUT_ROOT="outputs/roi_benchmark/${MODE}"
RESULTS_ROOT="${OUTPUT_ROOT}/evaluations"

if [[ "${MODE}" != "first" && "${MODE}" != "per-image" ]]; then
    printf 'Usage: %s first|per-image\n' "$0" >&2
    exit 2
fi

ROI_METHODS=(
    openai
    wide
    medium
    narrow
    left
    right
)

VIDEO_NAMES=(
    vidd_1
    vidd_2
    vidd_3
)

INPUT_DIRS=(
    data/benchmark_resized/vidd_1
    data/benchmark_resized/vidd_2
    data/benchmark_resized/vidd_3
)

REFERENCE_DIRS=(
    data/Labeled_Data/vidd_1/textfiles
    data/Labeled_Data/vidd_2/textfiles
    data/Labeled_Data/vidd_3/textfiles
)

for directory in "${INPUT_DIRS[@]}" "${REFERENCE_DIRS[@]}"; do
    if [[ ! -d "${directory}" ]]; then
        printf 'Missing benchmark directory: %s\n' "${directory}" >&2
        exit 1
    fi
done

mkdir -p "${RESULTS_ROOT}"

for roi_method in "${ROI_METHODS[@]}"; do
    output_method_root="${OUTPUT_ROOT}/${roi_method}"

    if [[ "${RUN_LABELING}" == "1" ]]; then
        for index in "${!VIDEO_NAMES[@]}"; do
            video_name="${VIDEO_NAMES[$index]}"
            input_dir="${INPUT_DIRS[$index]}"
            output_dir="${output_method_root}/${video_name}"

            printf '\n============================================================\n'
            printf 'Labeling ROI method: %s\n' "${roi_method}"
            printf 'Video: %s\n' "${video_name}"
            printf 'Mode: %s\n' "${MODE}"
            printf '============================================================\n'

            mkdir -p \
                "${output_dir}/annotations" \
                "${output_dir}/textfiles"

            "${PYTHON}" cli.py label \
                --input-dir "${input_dir}" \
                --output-images-dir "${output_dir}/annotations" \
                --output-coords-dir "${output_dir}/textfiles" \
                --roi-method "${roi_method}" \
                --roi-mode "${MODE}"
        done
    fi

    result_file="${RESULTS_ROOT}/${roi_method}.txt"

    printf '\n============================================================\n' | tee "${result_file}"
    printf 'Evaluating ROI method: %s\n' "${roi_method}" | tee -a "${result_file}"
    printf 'Mode: %s\n' "${MODE}" | tee -a "${result_file}"
    printf '============================================================\n' | tee -a "${result_file}"

    alina_dirs=()
    total_matches=0

    for index in "${!VIDEO_NAMES[@]}"; do
        video_name="${VIDEO_NAMES[$index]}"
        reference_dir="${REFERENCE_DIRS[$index]}"
        coords_dir="${output_method_root}/${video_name}/textfiles"

        if [[ ! -d "${coords_dir}" ]]; then
            printf 'Missing output directory: %s\n' "${coords_dir}" >&2
            exit 1
        fi

        matched_count=0
        for reference_file in "${reference_dir}"/*.txt; do
            [[ -e "${reference_file}" ]] || continue

            if [[ -f "${coords_dir}/$(basename -- "${reference_file}")" ]]; then
                ((matched_count += 1))
            fi
        done

        printf '%s: %d matching frame files\n' \
            "${video_name}" \
            "${matched_count}"

        ((total_matches += matched_count))
        alina_dirs+=("${coords_dir}")
    done

    if (( total_matches == 0 )); then
        printf 'No matching reference files for ROI method: %s\n' \
            "${roi_method}" >&2
        exit 1
    fi

    "${PYTHON}" cli.py evaluate \
        --canny-dirs \
            "${REFERENCE_DIRS[0]}" \
            "${REFERENCE_DIRS[1]}" \
            "${REFERENCE_DIRS[2]}" \
        --alina-dirs \
            "${alina_dirs[0]}" \
            "${alina_dirs[1]}" \
            "${alina_dirs[2]}" \
        | tee -a "${result_file}"
done

printf '\nBenchmark suite completed.\n'
printf 'Label outputs: %s\n' "${OUTPUT_ROOT}"
printf 'Evaluation results: %s\n' "${RESULTS_ROOT}"

