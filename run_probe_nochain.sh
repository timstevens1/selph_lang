#!/bin/bash
# §9.46 probe runner — NO CHAIN VARIANT.
# Same as run_probe.sh but does NOT load m_chain.selph. With
# __decomposers__ unbound, synth_v2's try_selph_decomposers returns
# nothing and only Flat enumeration runs. Used to confirm that the
# chain is the actual solver for the int control tasks.

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

COMBINED="$(mktemp -t probe_nochain.XXXXXX.selph)"
trap 'rm -f "$COMBINED"' EXIT

# Note: m_chain IS loaded (so __synth_skip__ gets populated and the
# M-stage helpers don't get probed as ~30 garbage-input library calls
# per task — the §9.45.13.3 fix). After m_chain loads, we override
# __decomposers__ to an empty namespace so try_selph_decomposers finds
# nothing and Flat is the actual solver. This isolates "did the chain
# solve it" from "is the chain present in env at all."
cat \
    "$META/m_pool.selph" \
    "$META/m13_data_atoms.selph" \
    "$META/m7_library_detection.selph" \
    "$META/m8_constant_fit.selph" \
    "$META/m9_unary_wrap.selph" \
    "$META/m10_affine_combination.selph" \
    "$META/m11_product_fit.selph" \
    "$META/m12_structural_pair_fit.selph" \
    "$META/m_chain.selph" \
    > "$COMBINED"

# Override __decomposers__ to empty AFTER m_chain registered the
# m-chain entry. try_selph_decomposers iterates this ns and calls each
# entry; an empty ns yields no matches.
echo '(define __decomposers__ (ns))' >> "$COMBINED"

cat "$CURRICULUM" >> "$COMBINED"

echo "Combined file (NO CHAIN): $COMBINED ($(wc -l < "$COMBINED") lines)" >&2
exec "$SELPH" grow-v2 "$COMBINED" "$@"
