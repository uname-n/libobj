#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

CC="${CC:-cc}"

echo "## build-smoke: building libobj (release)"
cargo build --release --quiet -p libobj

LIBOBJ_INCLUDE="$REPO_ROOT/crates/libobj/include"
LIBOBJ_DIR="$REPO_ROOT/target/release"

case "$(uname -s)" in
    Linux*)
        STATIC_LIB="$LIBOBJ_DIR/libobj.a"
        SYS_LIBS="-lpthread -ldl -lm"
        ;;
    Darwin*)
        STATIC_LIB="$LIBOBJ_DIR/libobj.a"
        SYS_LIBS="-framework CoreFoundation -framework Security -lpthread -lm"
        ;;
    *)
        echo "build-smoke: unsupported OS $(uname -s)" >&2
        exit 1
        ;;
esac

if [ ! -f "$STATIC_LIB" ]; then
    echo "build-smoke: expected libobj static lib at $STATIC_LIB" >&2
    exit 1
fi

SMOKE_SRC="$REPO_ROOT/crates/libobj/examples/smoke.c"
SMOKE_OUT="$REPO_ROOT/target/release/obj-c-smoke"

echo "## build-smoke: compiling smoke.c against $STATIC_LIB"
"$CC" -std=c99 -Wall -Wextra -Wpedantic \
    -I "$LIBOBJ_INCLUDE" \
    "$SMOKE_SRC" \
    "$STATIC_LIB" \
    $SYS_LIBS \
    -o "$SMOKE_OUT"

echo "## build-smoke: running $SMOKE_OUT"
OUTPUT="$("$SMOKE_OUT")"
echo "$OUTPUT"
if [ "$OUTPUT" != "OBJ_C_SMOKE_OK" ]; then
    echo "build-smoke: missing OBJ_C_SMOKE_OK marker; got: $OUTPUT" >&2
    exit 1
fi

echo "## build-smoke: OK"
