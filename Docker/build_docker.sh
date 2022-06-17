#!/bin/bash
DEVICE=auto

case "$1" in
--cuda) DEVICE=cuda ;;
--nocuda) DEVICE=nocuda ;;
--auto) DEVICE=auto ;;
esac
shift

docker build . -t dalle-mini:latest --build-arg device_type=$DEVICE