FILE=$1

if [ "$FILE" == "SRGAN_ImageNet" ]; then
  # Download the imagenet dataset and move validation images to labeled subfolders
  URL="https://huggingface.co/datasets/goodfellowliu/SRGAN_ImageNet/resolve/main/SRGAN_ImageNet.zip"
  ZIP_FILE=./data/SRGAN_ImageNet.zip
  mkdir -p ./data
  wget -N $URL -O $ZIP_FILE
  unzip $ZIP_FILE -d ./data
  rm $ZIP_FILE
elif [ "$FILE" == "Set5" ]; then
  # Download the Set5 dataset
  URL="https://huggingface.co/datasets/goodfellowliu/Set5/resolve/main/Set5.zip"
  ZIP_FILE=./data/Set5.zip
  mkdir -p ./data/Set5
  wget -N $URL -O $ZIP_FILE
  unzip $ZIP_FILE -d ./data/Set5
  rm $ZIP_FILE
else
  echo "Available arguments are SRGAN_ImageNet, Set5"
  exit 1
fi