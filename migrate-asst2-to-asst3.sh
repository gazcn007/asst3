#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <source-repo-path>"
  exit 1
fi

SRC="$1"
DST="."

FILES=(
  "src/cs248a_renderer/model/bvh.py"
  "src/cs248a_renderer/model/scene_object.py"
  "src/cs248a_renderer/slang_shaders/math/ray.slang"
  "src/cs248a_renderer/slang_shaders/math/bounding_box.slang"
  "src/cs248a_renderer/slang_shaders/model/bvh.slang"
  "src/cs248a_renderer/slang_shaders/model/camera.slang"
  "src/cs248a_renderer/slang_shaders/primitive/triangle.slang"
  "src/cs248a_renderer/slang_shaders/primitive/sdf.slang"
  "src/cs248a_renderer/slang_shaders/primitive/volume.slang"
  "src/cs248a_renderer/slang_shaders/renderer/volume_renderer.slang"
  "src/cs248a_renderer/slang_shaders/texture/diff_texture.slang"
  "src/cs248a_renderer/slang_shaders/texture/texture.slang"
)

for f in "${FILES[@]}"; do
  mkdir -p "$DST/$(dirname "$f")"
  cp "$SRC/$f" "$DST/$f"
  echo "Copied $f"
done

echo "Done."
