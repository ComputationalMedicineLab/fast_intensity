#!/bin/bash
set -e -x

# generate a builder script to include in the image
cat >builder <<'EOF'
#!/bin/bash
set -e -x

BUILD_REQS=(numpy cython)
VERSIONS=('35' '36' '37')

# Build the wheel
for v in "${VERSIONS[@]}"; do
    PIP=/opt/python/cp$v-cp${v}m/bin/pip
    $PIP install ${BUILD_REQS[@]}
    $PIP wheel /io/ -w wheelhouse
done

# Bundle the wheel's dependencies with it
for whl in wheelhouse/fast_intensity*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Cleanup not-our-wheels
shopt -s extglob
rm -f /io/wheelhouse/!(fast_intensity*.whl)
EOF
chmod +x builder

# pull and build the manylinux image
X86_IMG=quay.io/pypa/manylinux1_x86_64
docker pull $X86_IMG

# Run the builder, leaving wheels in ./wheelhouse
docker run --rm -v $(pwd):/io $X86_IMG /io/builder
# In case dist hasn't been generated
mkdir -p dist/
cp wheelhouse/* dist/

# Cleanup
rm builder
rm -rf wheelhouse
