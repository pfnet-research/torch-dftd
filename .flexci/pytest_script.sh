#!/bin/bash
set -eu

IMAGE=asia-northeast1-docker.pkg.dev/pfn-artifactregistry/torch-dftd/torch-dftd-ci:torch20
#IMAGE=torch-dftd-ci:torch20


main() {
  SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE}")/.."; pwd)"

  prepare_docker &
  wait

# Build and push docker images for unit tests.
#  bash -x -c "${SRC_ROOT}/.flexci/build_and_push.sh" \
#      "${IMAGE}"

# 1st pytest: when xdist is enabled with `-n $(nproc)`, benchmark is not executed.
# 2nd pytest: only execute pytest-benchmark.
  while :; do free -h; sleep 1; done &
  docker run --runtime=nvidia --rm --volume="$(pwd)":/workspace -w /workspace \
    ${IMAGE} \
    bash -x -c "pip install flake8 pytest pytest-cov pytest-xdist pytest-benchmark && \
      pip install cupy-cuda12x pytorch-pfn-extras!=0.5.0 && \
      pip install -e .[develop] && \
      pysen run lint && \
      pytest --cov=torch_dftd -n $(nproc) -m 'not slow' tests &&
      pytest --benchmark-only tests"
}


# prepare_docker makes docker use tmpfs to speed up.
# CAVEAT: Do not use docker during this is running.
prepare_docker() {
  service docker stop
  mount -t tmpfs -o size=100% tmpfs /var/lib/docker
  service docker start
  gcloud auth configure-docker
}


main "$@"
