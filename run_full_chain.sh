#!/bin/bash
# Full chained curriculum: sequence → CF → NL → arithmetic → CS
# Each domain loads only its own helpers + the accumulated library from prior stages.
# Traces are collected per-stage and merged at the end.

set -e

SELPH="./selph_fast/target/release/selph"
BUDGET="${BUDGET:-200000}"
OUT_DIR="chain_output"

mkdir -p "$OUT_DIR"

echo "=== SELPH Full Chained Curriculum ==="
echo "Budget: $BUDGET"
echo ""

# Stage 1: Sequence tasks (13 tasks)
# List inputs — no seq_helpers needed, type filtering excludes string ops
echo "=== Stage 1: Sequence (13 tasks) ==="
$SELPH grow examples/sequence_tasks_list.selph \
    --output "$OUT_DIR/stage1_seq.selph" \
    --meta \
    --trace "$OUT_DIR/trace_seq.json" \
    --budget "$BUDGET"

echo ""

# Stage 2: Context-free (20 tasks)
# Loads cf_helpers + stage1 library (sequence solutions)
echo "=== Stage 2: Context-Free (20 tasks) ==="
$SELPH grow examples/context_free_tasks.selph \
    --library examples/cf_helpers.selph \
    --library "$OUT_DIR/stage1_seq.selph" \
    --output "$OUT_DIR/stage2_cf.selph" \
    --meta \
    --trace "$OUT_DIR/trace_cf.json" \
    --budget "$BUDGET"

echo ""

# Stage 3: Natural language (22 tasks)
# Loads nl_helpers + stage2 library (seq + CF solutions)
echo "=== Stage 3: Natural Language (22 tasks) ==="
$SELPH grow examples/nl_tasks.selph \
    --library examples/nl_helpers.selph \
    --library "$OUT_DIR/stage2_cf.selph" \
    --output "$OUT_DIR/stage3_nl.selph" \
    --meta \
    --trace "$OUT_DIR/trace_nl.json" \
    --budget "$BUDGET"

echo ""

# Stage 4: Integer arithmetic (2 tasks)
# Teaches divide/floor compositions — no domain helpers needed
echo "=== Stage 4: Arithmetic (2 tasks) ==="
$SELPH grow examples/arithmetic_tasks.selph \
    --output "$OUT_DIR/stage4_arith.selph" \
    --meta \
    --trace "$OUT_DIR/trace_arith.json" \
    --budget "$BUDGET"

echo ""

# Stage 5: String slicing (8 tasks)
# Loads stage4 arithmetic library (halve, third)
echo "=== Stage 5: String Slicing (8 tasks) ==="
$SELPH grow examples/string_slice_tasks.selph \
    --library "$OUT_DIR/stage4_arith.selph" \
    --output "$OUT_DIR/stage5_slice.selph" \
    --meta \
    --trace "$OUT_DIR/trace_slice.json" \
    --budget "$BUDGET"

echo ""

# Stage 6: Context-sensitive (11 tasks)
# Loads cf_helpers + cs_helpers + stage2 CF library + stage5 slicing library
echo "=== Stage 6: Context-Sensitive (11 tasks) ==="
$SELPH grow examples/cs_tasks.selph \
    --library examples/cf_helpers.selph \
    --library examples/cs_helpers.selph \
    --library "$OUT_DIR/stage2_cf.selph" \
    --library "$OUT_DIR/stage5_slice.selph" \
    --output "$OUT_DIR/stage6_cs.selph" \
    --meta \
    --trace "$OUT_DIR/trace_cs.json" \
    --budget "$BUDGET"

echo ""
echo "=== Chain complete ==="
echo "Stage outputs: $OUT_DIR/stage{1,2,3,4,5,6}_*.selph"
echo "Traces: $OUT_DIR/trace_{seq,cf,nl,arith,slice,cs}.json"
echo "Final library: $OUT_DIR/stage6_cs.selph"
