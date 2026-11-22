#!/usr/bin/env bash
# conda-forge libtorch 2.3.1 ships libtorch_cpu.so with an executable-stack flag (PT_GNU_STACK RWE).
# glibc >= 2.41 refuses to dlopen such objects ("cannot enable executable stack as shared object
# requires: Invalid argument"), which breaks `import torch` on current GitHub runners and new distros.
# Clearing the flag is safe: nothing in libtorch needs an executable stack. Run once after creating
# the environment (CI and the SageMaker image do this automatically).
#
#   scripts/fix_execstack.sh            # patches the active python's site-packages
#   scripts/fix_execstack.sh /path/to/env
set -euo pipefail
PREFIX="${1:-${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}}"
PY="$PREFIX/bin/python"
"$PY" -m pip install --quiet --disable-pip-version-check patchelf
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
patched=0
has_execstack() {  # minimal images (micromamba) have no binutils; fall back to the known offender
  if command -v readelf >/dev/null 2>&1; then
    readelf -lW "$1" 2>/dev/null | grep -q 'GNU_STACK.*RWE'
  else
    [[ "$(basename "$1")" == libtorch_cpu.so* ]]
  fi
}
while IFS= read -r -d '' f; do
  if has_execstack "$f"; then
    "$PREFIX/bin/patchelf" --clear-execstack "$f"
    echo "cleared executable stack: $f"
    patched=$((patched + 1))
  fi
done < <(find "$SITE/torch/lib" "$SITE/dgl" "$PREFIX/lib" -maxdepth 2 -type f -name '*.so*' -print0 2>/dev/null)
echo "patched $patched shared objects"
"$PY" -c 'import torch, dgl; print("import ok:", torch.__version__, dgl.__version__)'
