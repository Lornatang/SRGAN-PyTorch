# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# ================================================ ===========================
# File description: Realize the model training function.
# ================================================ ===========================
from torch.utils.data import DataLoader

from config import *
from dataset import BaseDataset


def train(train_dataloader, epoch) -> None:
    """The purpose of training is to minimize the pixel difference between super-resolution images and high-resolution images."""
    batches = len(train_dataloader)

    # Start training mode.
    model.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        # 0. Copy the data to the designated device.
        lr = lr.to(device)
        hr = hr.to(device)
        # 1. Set the gradient of the generated model to 0.
        model.zero_grad()
        # 2. Calculate the difference between the super-resolution image and the high-resolution image at the pixel level.
        sr = model(lr)
        pixel_loss = criterion(sr, hr)
        # 3. Update the weights of the generative model.
        pixel_loss.backward()
        optimizer.step()

        # Write the loss value in the process of training and generating the network into Tensorboard.
        iters = index + epoch * batches + 1
        writer.add_scalar("Train/Loss", pixel_loss.item(), iters)
        # Only output the last loss function.
        if (index + 1) == batches:
            print(f"Epoch[{epoch + 1:05d}/{epochs:05d}]({index + 1:05d}/{batches:05d}) "
                  f"Loss: {pixel_loss.item():.6f}.")


def validate(valid_dataloader, epoch) -> float:
    """Verify the performance of the model after each Epoch."""
    batches = len(valid_dataloader)
    total_psnr_value = 0.0

    # Start verification mode.
    model.eval()

    with torch.no_grad():
        for index, (lr, hr) in enumerate(valid_dataloader):
            # Copy the data to the specified device.
            lr = lr.to(device)
            hr = hr.to(device)
            # Calculate the loss.
            sr = model(lr)
            # Calculate the PSNR indicator.
            mse_loss = ((sr - hr) ** 2).data.mean()
            psnr_value = 10 * torch.log10(1 / mse_loss).item()
            total_psnr_value += psnr_value

        avg_psnr_value = total_psnr_value / batches
        # Write the value of each round of verification indicators into Tensorboard.
        writer.add_scalar("Valid/PSNR", avg_psnr_value, epoch + 1)
        # Print evaluation indicators.
        print(f"Test Epoch[{epoch + 1:05d}] avg PSNR: {avg_psnr_value:.2f}.")

    return avg_psnr_value


def main() -> None:
    # Create a super-resolution experiment result folder.
    if not os.path.exists(exp_dir1):
        os.makedirs(exp_dir1)
    if not os.path.exists(exp_dir2):
        os.makedirs(exp_dir2)

    # Load the data set.
    train_dataset = BaseDataset(train_dir)
    valid_dataset = BaseDataset(valid_dir)
    train_dataloader = DataLoader(train_dataset, batch_size, True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, False, pin_memory=True)
    # Check whether the training progress of the last abnormal end is restored, for example, the power is cut off in the middle of the training.
    if resume:
        print("Resuming...")
        model.load_state_dict(torch.load(resume_weight))

    # Initialize the evaluation index of the training phase
    best_psnr_value = 0.0
    for epoch in range(start_epoch, epochs):
        # Train the generator model under each Epoch.
        train(train_dataloader, epoch)
        # Verify the model performance after each Epoch.
        psnr_value = validate(valid_dataloader, epoch)
        # Evaluate whether the performance of the current node model is the highest indicator.
        is_best = psnr_value > best_psnr_value
        best_psnr_value = max(psnr_value, best_psnr_value)
        # Save the node model. If the current node model has the best performance, a model file ending with `-best` will also be saved.
        torch.save(model.state_dict(), os.path.join(exp_dir1, f"epoch{epoch + 1}.pth"))
        if is_best:
            torch.save(model.state_dict(), os.path.join(exp_dir2, "best.pth"))

    # Save the last iteration training model.
    torch.save(model.state_dict(), os.path.join(exp_dir2, "last.pth"))


if __name__ == "__main__":
    main()
