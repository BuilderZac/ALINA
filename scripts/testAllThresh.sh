#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"
ROI_METHOD="${ROI_METHOD:-medium_modded}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/threshold_benchmark}"
RUN_LABELING="${RUN_LABELING:-1}"

RANDOM_FOREST_MODEL="${RANDOM_FOREST_MODEL:-${PROJECT_ROOT}/random_forest.joblib}"
XGBOOST_MODEL="${XGBOOST_MODEL:-${PROJECT_ROOT}/xgboost.joblib}"


THRESHOLD_METHODS=(
    manual
    kmeans
    gmm
    random_forest
    xgboost
    openai
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

mkdir -p "${OUTPUT_ROOT}"

for threshold_method in "${THRESHOLD_METHODS[@]}"; do
    method_root="${OUTPUT_ROOT}/${threshold_method}"
    result_file="${method_root}/evaluation.txt"

    mkdir -p "${method_root}"

    if [[ "${RUN_LABELING}" == "1" ]]; then
        for index in "${!VIDEO_NAMES[@]}"; do
            video_name="${VIDEO_NAMES[$index]}"
            input_dir="${INPUT_DIRS[$index]}"
            output_dir="${method_root}/${video_name}"

            printf '\n============================================================\n'
            printf 'Threshold method: %s\n' "${threshold_method}"
            printf 'Fixed ROI method: %s\n' "${ROI_METHOD}"
            printf 'Video: %s\n' "${video_name}"
            printf '============================================================\n'

            mkdir -p \
                "${output_dir}/annotations" \
                "${output_dir}/textfiles"

            threshold_model_args=()

            case "${threshold_method}" in
                random_forest)
                    threshold_model_args=(
                        --threshold-model-path
                        "${RANDOM_FOREST_MODEL}"
                    )
                    ;;
                xgboost)
                    threshold_model_args=(
                        --threshold-model-path
                        "${XGBOOST_MODEL}"
                    )
                    ;;
            esac

            "${PYTHON}" cli.py label \
                --input-dir "${input_dir}" \
                --output-images-dir "${output_dir}/annotations" \
                --output-coords-dir "${output_dir}/textfiles" \
                --roi-method "${ROI_METHOD}" \
                --roi-mode first \
                --threshold-method "${threshold_method}" \
                --log-file "${output_dir}/timing.log" \
                "${threshold_model_args[@]}"
        done
    fi

    printf '\n============================================================\n' \
        | tee "${result_file}"
    printf 'Threshold method: %s\n' "${threshold_method}" \
        | tee -a "${result_file}"
    printf 'Fixed ROI method: %s\n' "${ROI_METHOD}" \
        | tee -a "${result_file}"
    printf '============================================================\n' \
        | tee -a "${result_file}"

    alina_dirs=()

    for index in "${!VIDEO_NAMES[@]}"; do
        video_name="${VIDEO_NAMES[$index]}"
        coords_dir="${method_root}/${video_name}/textfiles"

        if [[ ! -d "${coords_dir}" ]]; then
            printf 'Missing output directory: %s\n' "${coords_dir}" >&2
            exit 1
        fi

        alina_dirs+=("${coords_dir}")
    done

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

printf '\nThreshold benchmark completed.\n'
printf 'Results: %s\n' "${OUTPUT_ROOT}"

