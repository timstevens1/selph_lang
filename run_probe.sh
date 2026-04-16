#!/bin/bash
# §9.46 probe runner: concatenates the M-chain in dependency order
# in front of the requested curriculum file and runs `selph grow-v2`.
#
# Usage: ./run_probe.sh <curriculum.selph> [extra grow-v2 args ...]
#
# Why this exists: grow-v2 has no --library or --preamble flag, so the
# pure-SELPH M-stage chain has to be physically prepended to whatever
# task file we want to run against. This script does that consistently
# so probes are reproducible.

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

# Dependency-ordered load. m_pool first (provides make-pool); m13
# (provides extract-data-atoms which m_pool's "constants" flag and
# m11/m12 depend on); then m7..m12 in any order; m_chain LAST because
# it registers __decomposers__ and __synth_skip__ from current env.
COMBINED="$(mktemp -t probe.XXXXXX.selph)"
trap 'rm -f "$COMBINED"' EXIT

# List of files in concat order (for offset tracking)
FILES=(
    "$META/m_pool.selph"
    "$META/m13_data_atoms.selph"
    "$META/m7_library_detection.selph"
    "$META/m8_constant_fit.selph"
    "$META/m9_unary_wrap.selph"
    "$META/m10_affine_combination.selph"
    "$META/m11_product_fit.selph"
    "$META/m12_structural_pair_fit.selph"
    "$META/m_pool_string.selph"
    "$META/m8s_constant_string.selph"
    "$META/m10s_concat_pair.selph"
    "$META/m11s_string_repeat.selph"
    "$META/m_pool_grid.selph"
    "$META/m8g_constant_grid.selph"
    "$META/m8g_symmetry.selph"
    "$META/m8g_recolor.selph"
    "$META/m8g_line_draw.selph"
    "$META/m8g_proximity_recolor.selph"
    "$META/m8g_rect_hole_fill.selph"
    "$META/m8g_per_object.selph"
    "$META/m8g_mono_object.selph"
    "$META/m8g_template_stamp.selph"
    "$META/m8g_compose.selph"
    "$META/m_journal.selph"
    "$META/m_chain.selph"
    "$META/m_partition.selph"
    "$META/m_lib_reuse.selph"
    "$META/m_fitness_grid.selph"
    "$META/m_refine.selph"
    "$META/m_ho.selph"
    "$META/m_ho_list_map.selph"
    "$META/m_ho_list_filter.selph"
    "$META/m_ho_split_map_join.selph"
    "$META/m_ho_char_map_join.selph"
    "$META/m_rd.selph"
    "$META/m_dc.selph"
    "$CURRICULUM"
)

# Concatenate files and track offsets/line counts
OFFSET=0
echo "File offsets in combined file:" >&2
for f in "${FILES[@]}"; do
    LINES=$(wc -l < "$f")
    echo "  $f: offset=$OFFSET, lines=$LINES" >&2
    OFFSET=$((OFFSET + LINES))
done
cat "${FILES[@]}" > "$COMBINED"

echo "Combined file: $COMBINED ($OFFSET lines total)" >&2
exec "$SELPH" grow-v2 "$COMBINED" "$@"
