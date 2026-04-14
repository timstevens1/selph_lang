#!/bin/bash
# §9.55 fitness probe runner: like run_probe.sh but with fitness-aware
# post-mortem that retries failures with the M-chain + boost heuristic.
#
# Usage: ./run_fitness_probe.sh <curriculum.selph> [grow-v2 args ...]

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <curriculum.selph> [grow-v2 args]" >&2
    exit 1
fi

CURRICULUM="$1"
shift

REPO="$(cd "$(dirname "$0")" && pwd)"
SELPH="$REPO/selph_fast/target/release/selph"
META="$REPO/examples/meta_curriculum"

if [ ! -x "$SELPH" ]; then
    echo "Building selph..." >&2
    (cd "$REPO/selph_fast" && cargo build --release >&2)
fi

# Build combined curriculum (M-chain + tasks)
COMBINED="$(mktemp -t probe.XXXXXX.selph)"
trap 'rm -f "$COMBINED" "$PM_COMBINED"' EXIT

cat \
    "$META/m_pool.selph" \
    "$META/m13_data_atoms.selph" \
    "$META/m7_library_detection.selph" \
    "$META/m8_constant_fit.selph" \
    "$META/m9_unary_wrap.selph" \
    "$META/m10_affine_combination.selph" \
    "$META/m11_product_fit.selph" \
    "$META/m12_structural_pair_fit.selph" \
    "$META/m_pool_string.selph" \
    "$META/m8s_constant_string.selph" \
    "$META/m10s_concat_pair.selph" \
    "$META/m11s_string_repeat.selph" \
    "$META/m_pool_grid.selph" \
    "$META/m8g_constant_grid.selph" \
    "$META/m8g_symmetry.selph" \
    "$META/m8g_recolor.selph" \
    "$META/m8g_line_draw.selph" \
    "$META/m8g_per_object.selph" \
    "$META/m8g_template_stamp.selph" \
    "$META/m8g_compose.selph" \
    "$META/m_chain.selph" \
    "$META/m_fitness_grid.selph" \
    "$META/m_refine.selph" \
    "$META/m_ho.selph" \
    "$CURRICULUM" \
    > "$COMBINED"

# Build post-mortem with M-chain + fitness modules (for retry)
PM_COMBINED="$(mktemp -t pm.XXXXXX.selph)"
cat \
    "$META/m_pool.selph" \
    "$META/m13_data_atoms.selph" \
    "$META/m7_library_detection.selph" \
    "$META/m8_constant_fit.selph" \
    "$META/m9_unary_wrap.selph" \
    "$META/m10_affine_combination.selph" \
    "$META/m11_product_fit.selph" \
    "$META/m12_structural_pair_fit.selph" \
    "$META/m_pool_string.selph" \
    "$META/m8s_constant_string.selph" \
    "$META/m10s_concat_pair.selph" \
    "$META/m11s_string_repeat.selph" \
    "$META/m_pool_grid.selph" \
    "$META/m8g_constant_grid.selph" \
    "$META/m8g_symmetry.selph" \
    "$META/m8g_recolor.selph" \
    "$META/m8g_line_draw.selph" \
    "$META/m8g_per_object.selph" \
    "$META/m8g_template_stamp.selph" \
    "$META/m8g_compose.selph" \
    "$META/m_chain.selph" \
    "$META/m_fitness_grid.selph" \
    "$META/m_refine.selph" \
    "$META/m_ho.selph" \
    "$META/m_near_miss.selph" \
    "$META/m_boost_heuristic.selph" \
    "$META/m_phase3_compose.selph" \
    "$META/m_phase4_templates.selph" \
    > "$PM_COMBINED"
echo '(define run-post-mortem (lambda (results) (run-fitness-post-mortem results)))' >> "$PM_COMBINED"

echo "Combined curriculum: $(wc -l < "$COMBINED") lines" >&2
echo "Post-mortem: $(wc -l < "$PM_COMBINED") lines" >&2
exec "$SELPH" grow-v2 "$COMBINED" --post-mortem "$PM_COMBINED" "$@"
