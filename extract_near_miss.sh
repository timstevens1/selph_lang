#!/bin/bash
# Extract near-miss tasks (fitness >= threshold) from a probe run's output.
# Generates a curriculum file containing only those tasks.
#
# Usage: ./extract_near_miss.sh <probe_output.txt> <full_curriculum.selph> [threshold]
#
# The probe output must contain lines like:
#   [N/M] FAIL  taskname  N cand  fitness=0.933
#
# Default threshold: 0.50 (captures near-miss + partial)

set -e

if [ -z "$2" ]; then
    echo "Usage: $0 <probe_output.txt> <curriculum.selph> [threshold=0.50] [output.selph]" >&2
    exit 1
fi

PROBE_OUTPUT="$1"
CURRICULUM="$2"
THRESHOLD="${3:-0.50}"
OUTPUT="${4:-/tmp/arc_near_miss.selph}"

# Extract task names with fitness >= threshold
echo "Extracting tasks with fitness >= $THRESHOLD from $PROBE_OUTPUT" >&2
TASK_IDS=$(grep -oP 'FAIL\s+\K\S+(?=\s.*fitness=)' "$PROBE_OUTPUT" | while read -r name; do
    fitness=$(grep -oP "FAIL\s+${name}\s.*fitness=\K[0-9.]+" "$PROBE_OUTPUT" | head -1)
    if [ -n "$fitness" ] && python3 -c "exit(0 if float('$fitness') >= float('$THRESHOLD') else 1)" 2>/dev/null; then
        echo "$name"
    fi
done)

COUNT=$(echo "$TASK_IDS" | grep -c . || true)
echo "Found $COUNT tasks with fitness >= $THRESHOLD" >&2

# Extract those tasks from the curriculum file
echo ";; Near-miss ARC tasks (fitness >= $THRESHOLD)" > "$OUTPUT"
echo ";; Extracted from $PROBE_OUTPUT" >> "$OUTPUT"
echo "" >> "$OUTPUT"

for TASK_ID in $TASK_IDS; do
    # Extract the task block: from ";; Task: ID" to the next ";; Task:" or EOF
    awk -v id="$TASK_ID" '
        /^;; Task: / { printing = ($3 == id) }
        printing { print }
    ' "$CURRICULUM" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

EXTRACTED=$(grep -c '^(task-args' "$OUTPUT" || true)
echo "Wrote $EXTRACTED tasks to $OUTPUT" >&2
echo "$OUTPUT"
