#!/bin/bash -uex

IMAGE_BASE="${1:-}"
IMAGE_PUSH=1
if [ "${IMAGE_BASE}" = "" ]; then
  IMAGE_BASE="torch-dftd"
  IMAGE_PUSH=0
fi

TEST_PIP_PACKAGES="
flake8 pytest pytest-cov pytest-xdist pytest-benchmark
"

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE}")/.."; pwd)"

docker_build_and_push() {

  IMAGE_TAG="${1}"; shift
  IMAGE_NAME="${IMAGE_BASE}:${IMAGE_TAG}"

  pushd "$(dirname ${0})"
  docker build -f ${SRC_ROOT}/docker/Dockerfile -t "${IMAGE_NAME}" "$@" .
  popd

  if [ "${IMAGE_PUSH}" = "0" ]; then
    echo "Skipping docker push."
  else
    docker push "${IMAGE_NAME}"
  fi
}

WAIT_PIDS=""

# PyTorch 1.5 + Python 3.6
docker_build_and_push torch15 \
    --build-arg base_image="nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04" \
    --build-arg python_version="3.6.12" \
    --build-arg pip_packages="torch==1.5.* torchvision==0.6.* ${TEST_PIP_PACKAGES}" &
WAIT_PIDS="$! ${WAIT_PIDS}"

# Wait until the build complete.
for P in ${WAIT_PIDS}; do
    wait ${P}
done
