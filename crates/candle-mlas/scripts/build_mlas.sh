#!/bin/sh
set -eu
sdk_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
root=${MLAS_WORK_DIR:-"$sdk_root"}
archive="$root/vendor/onnxruntime-v1.30.0.tar.gz"
source_root="$root/vendor/onnxruntime"
mkdir -p "$root/vendor" "$root/results"
if [ ! -f "$archive" ]; then
  curl -fLsS https://codeload.github.com/microsoft/onnxruntime/tar.gz/f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7 -o "$archive.partial"
  mv "$archive.partial" "$archive"
fi
printf '%s  %s\n' 1b4ec55fa26ee8fe622ea256894dfe9381bdfa99d41489994a1961720f556c48 "$archive" | sha256sum -c -
# Recreate derived sources on every build from the verified archive. A cached
# CMake file or modified extracted tree is never treated as provenance evidence.
rm -rf -- "$source_root"
mkdir -p "$source_root"
tar -xzf "$archive" --strip-components=1 -C "$source_root"
python3 - "$root" <<'PY'
import hashlib, shutil, sys, urllib.request, zipfile
from pathlib import Path
root=Path(sys.argv[1])
for name,url,expected in [('safeint','https://github.com/dcleblanc/SafeInt/archive/refs/tags/3.0.28.zip','23f252040ff6cb9f1fd18575b32fa8fb5928daac'),('gsl','https://github.com/microsoft/GSL/archive/refs/tags/v4.2.1.zip','1094e3bb7a8af763dcb136ccd676e6e75e614eec')]:
    archive=root/'vendor'/(name+'.zip')
    if not archive.exists():
        archive.write_bytes(urllib.request.urlopen(url,timeout=90).read())
    assert hashlib.sha1(archive.read_bytes()).hexdigest()==expected,(name,'upstream deps.txt SHA1 mismatch')
    dest=root/'vendor'/name
    if dest.is_symlink():dest.unlink()
    elif dest.exists():shutil.rmtree(dest)
    dest.mkdir()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            relative=Path(*Path(item.filename).parts[1:])
            if item.is_dir() or not relative.parts:continue
            assert '..' not in relative.parts
            p=dest/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(item))
PY
# Clean objects too: restored archive timestamps must not permit an object built
# from previously modified sources to be reused.
rm -rf -- "$root/vendor/mlas-build"
cmake -S "$sdk_root/native" -B "$root/vendor/mlas-build" -DCMAKE_BUILD_TYPE=Release -DMLAS_TARGET_NATIVE="${MLAS_TARGET_NATIVE:-OFF}" -DVELA_KERNEL_SOURCE="${VELA_KERNEL_SOURCE:-}" -DORT_SOURCE="$source_root" -DSAFEINT_SOURCE="$root/vendor/safeint" -DGSL_SOURCE="$root/vendor/gsl"
cmake --build "$root/vendor/mlas-build" -j 4
sha256sum "$root/vendor/mlas-build/libcandle_mlas.so"
if [ -n "${VELA_KERNEL_SOURCE:-}" ]; then sha256sum "$root/vendor/mlas-build/libvela_mlas.so"; fi
