DEVICE=auto

case "$1" in
--cpu) DEVICE=cpu ;;
--gpu) DEVICE=gpu ;;
--auto) DEVICE=auto ;;
esac
shift

docker build . -t dalle-mini:latest --build-arg device_type=$DEVICE