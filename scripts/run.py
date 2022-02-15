import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/ImageNet/original --output_dir ../data/ImageNet/SRGAN/train --image_size 96 --step 48 --num_workers 10")
os.system("python ./split_train_valid_dataset.py --train_images_dir ../data/ImageNet/SRGAN/train --valid_images_dir ../data/ImageNet/SRGAN/valid --valid_samples_ratio 0.1")

# Create LMDB database file
# os.system("python ./create_lmdb_dataset.py --images_dir ../data/ImageNet/SRGAN/train --lmdb_path ../data/train_lmdb/SRGAN/ImageNet_HR_lmdb --upscale_factor 1")
# os.system("python ./create_lmdb_dataset.py --images_dir ../data/ImageNet/SRGAN/train --lmdb_path ../data/train_lmdb/SRGAN/ImageNet_LRbicx4_lmdb --upscale_factor 4")

# os.system("python ./create_lmdb_dataset.py --images_dir ../data/ImageNet/SRGAN/valid --lmdb_path ../data/valid_lmdb/SRGAN/ImageNet_HR_lmdb --upscale_factor 1")
# os.system("python ./create_lmdb_dataset.py --images_dir ../data/ImageNet/SRGAN/valid --lmdb_path ../data/valid_lmdb/SRGAN/ImageNet_LRbicx4_lmdb --upscale_factor 4")
