FILE=$1

if [ "$FILE" == "SRGAN_x2-ImageNet" ]; then
  URL="https://huggingface.co/ChangyuLiu/SRGAN-PyTorch/resolve/main/SRGAN_x2-ImageNet.pth.tar"
  FILE_PATH=./results/pretrained_models
  wget $URL -P $FILE_PATH
elif [ "$FILE" == "SRGAN_x4-ImageNet" ]; then
  URL="https://huggingface.co/ChangyuLiu/SRGAN-PyTorch/resolve/main/SRGAN_x4-ImageNet.pth.tar"
  FILE_PATH=./results/pretrained_models
  wget $URL -P $FILE_PATH
elif [ "$FILE" == "SRGAN_x8-ImageNet" ]; then
  URL="https://huggingface.co/ChangyuLiu/SRGAN-PyTorch/resolve/main/SRGAN_x8-ImageNet.pth.tar"
  FILE_PATH=./results/pretrained_models
  wget $URL -P $FILE_PATH
elif [ "$FILE" == "SRResNet_x2-ImageNet" ]; then
  URL="https://huggingface.co/ChangyuLiu/SRGAN-PyTorch/resolve/main/SRResNet_x2-ImageNet.pth.tar"
  FILE_PATH=./results/pretrained_models
  wget $URL -P $FILE_PATH
elif [ "$FILE" == "SRResNet_x4-ImageNet" ]; then
  URL="https://huggingface.co/ChangyuLiu/SRGAN-PyTorch/resolve/main/SRResNet_x4-ImageNet.pth.tar"
  FILE_PATH=./results/pretrained_models
  wget $URL -P $FILE_PATH
elif [ "$FILE" == "SRResNet_x8-ImageNet" ]; then
  URL="https://huggingface.co/ChangyuLiu/SRGAN-PyTorch/resolve/main/SRResNet_x8-ImageNet.pth.tar"
  FILE_PATH=./results/pretrained_models
  wget $URL -P $FILE_PATH
elif [ "$FILE" == "DiscriminatorForVGG_x2-ImageNet" ]; then
  URL="https://huggingface.co/ChangyuLiu/SRGAN-PyTorch/resolve/main/DiscriminatorForVGG_x2-ImageNet.pth.tar"
  FILE_PATH=./results/pretrained_models
  wget $URL -P $FILE_PATH
elif [ "$FILE" == "DiscriminatorForVGG_x4-ImageNet" ]; then
  URL="https://huggingface.co/ChangyuLiu/SRGAN-PyTorch/resolve/main/DiscriminatorForVGG_x4-ImageNet.pth.tar"
  FILE_PATH=./results/pretrained_models
  wget $URL -P $FILE_PATH
elif [ "$FILE" == "DiscriminatorForVGG_x8-ImageNet" ]; then
  URL="https://huggingface.co/ChangyuLiu/SRGAN-PyTorch/resolve/main/DiscriminatorForVGG_x8-ImageNet.pth.tar"
  FILE_PATH=./results/pretrained_models
  wget $URL -P $FILE_PATH
else
  echo "Available arguments are SRGAN_x2-ImageNet, SRGAN_x4-ImageNet, SRGAN_x8-ImageNet, SRResNet_x2-ImageNet, SRResNet_x4-ImageNet, SRResNet_x8-ImageNet, DiscriminatorForVGG_x2-ImageNet, DiscriminatorForVGG_x4-ImageNet, DiscriminatorForVGG_x8-ImageNet"
  exit 1
fi
