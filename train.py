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

# ============================================================================
# File description: Realize the model training function.
# ============================================================================
from torch.utils.data import DataLoader

from config import *
from dataset import BaseDataset


def train_generator(train_dataloader, epoch) -> None:
    """Only train the generative model.

    Args:
        train_dataloader (torch.utils.data.DataLoader): The loader of the training dataset.
        epoch (int): number of training cycles.
    """
    # Calculate how many iterations there are under epoch.
    batches = len(train_dataloader)
    # Set generator network in training mode.
    generator.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        # Copy the data to the specified device.
        lr = lr.to(device)
        hr = hr.to(device)
        # Initialize the gradient of the generator model.
        generator.zero_grad()
        # Generate super-resolution images.
        sr = generator(lr)
        # Calculate the difference between the super-resolution image and the high-resolution image at the pixel level.
        pixel_loss = pixel_criterion(sr, hr)
        # Update the weights of the generator model.
        pixel_loss.backward()
        p_optimizer.step()
        # Write the loss during training into Tensorboard.
        iters = index + epoch * batches + 1
        writer.add_scalar("Train_Generator/Loss", pixel_loss.item(), iters)
        # Print the loss function every ten iterations and the last iteration in this epoch.
        if (index + 1) % 10 == 0 or (index + 1) == batches:
            print(f"Train Epoch[{epoch + 1:04d}/{p_epochs:04d}]({index + 1:05d}/{batches:05d}) "
                  f"Loss: {pixel_loss.item():.6f}.")


def train_adversarial(train_dataloader, epoch) -> None:
    """Training generative models and adversarial models.

    Args:
        train_dataloader (torch.utils.data.DataLoader): The loader of the training dataset.
        epoch (int): number of training cycles.
    """
    # Calculate how many iterations there are under Epoch.
    batches = len(train_dataloader)
    # Set adversarial network in training mode.
    discriminator.train()
    generator.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        # Copy the data to the specified device.
        lr = lr.to(device)
        hr = hr.to(device)
        label_size = lr.size(0)
        # Create label. Set the real sample label to 1, and the false sample label to 0.
        real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, device=device)
        fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, device=device)

        # Initialize the gradient of the discriminator model.
        discriminator.zero_grad()
        # Calculate the loss of the discriminator model on the high-resolution image.
        output = discriminator(hr)
        d_loss_hr = adversarial_criterion(output, real_label)
        d_loss_hr.backward()
        d_hr = output.mean().item()
        # Generate super-resolution images.
        sr = generator(lr)
        # Calculate the loss of the discriminator model on the super-resolution image.
        output = discriminator(sr.detach())
        d_loss_sr = adversarial_criterion(output, fake_label)
        d_loss_sr.backward()
        d_sr1 = output.mean().item()
        # Update the weights of the discriminator model.
        d_loss = d_loss_hr + d_loss_sr
        d_optimizer.step()

        # Initialize the gradient of the generator model.
        generator.zero_grad()
        # Calculate the loss of the discriminator model on the super-resolution image.
        output = discriminator(sr)
        # Perceptual loss=0.01 * pixel loss + 1.0 * content loss + 0.001 * adversarial loss.
        pixel_loss = pixel_weight * pixel_criterion(sr, hr.detach())
        perceptual_loss = content_weight * content_criterion(sr, hr.detach())
        adversarial_loss = adversarial_weight * adversarial_criterion(output, real_label)
        # Update the weights of the generator model.
        g_loss = pixel_loss + perceptual_loss + adversarial_loss
        g_loss.backward()
        g_optimizer.step()
        d_sr2 = output.mean().item()

        # Write the loss during training into Tensorboard.
        iters = index + epoch * batches + 1
        writer.add_scalar("Train_Adversarial/D_Loss", d_loss.item(), iters)
        writer.add_scalar("Train_Adversarial/G_Loss", g_loss.item(), iters)
        writer.add_scalar("Train_Adversarial/D_HR", d_hr, iters)
        writer.add_scalar("Train_Adversarial/D_SR1", d_sr1, iters)
        writer.add_scalar("Train_Adversarial/D_SR2", d_sr2, iters)
        # Print the loss function every ten iterations and the last iteration in this Epoch.
        if (index + 1) % 10 == 0 or (index + 1) == batches:
            print(f"Train stage: adversarial "
                  f"Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                  f"D Loss: {d_loss.item():.6f} G Loss: {g_loss.item():.6f} "
                  f"D(HR): {d_hr:.6f} D(SR1)/D(SR2): {d_sr1:.6f}/{d_sr2:.6f}.")


def validate(valid_dataloader, epoch, stage) -> float:
    """Verify the generative model.

    Args:
        valid_dataloader (torch.utils.data.DataLoader): loader for validating dataset.
        epoch (int): number of training cycles.
        stage (str): In which stage to verify, one is `generator`, the other is `adversarial`.

    Returns:
        PSNR value(float).
    """
    # Calculate how many iterations there are under epoch.
    batches = len(valid_dataloader)
    # Set generator model in verification mode.
    generator.eval()
    # Initialize the evaluation index.
    total_psnr_value = 0.0

    with torch.no_grad():
        for index, (lr, hr) in enumerate(valid_dataloader):
            # Copy the data to the specified device.
            lr = lr.to(device)
            hr = hr.to(device)
            # Generate super-resolution images.
            sr = generator(lr)
            # Calculate the PSNR indicator.
            mse_loss = psnr_criterion(sr, hr)
            psnr_value = 10 * torch.log10(1 / mse_loss).item()
            total_psnr_value += psnr_value

        avg_psnr_value = total_psnr_value / batches
        # Write the value of each round of verification indicators into Tensorboard.
        if stage == "generator":
            writer.add_scalar("Val_Generator/PSNR", avg_psnr_value, epoch + 1)
        elif stage == "adversarial":
            writer.add_scalar("Val_Adversarial/PSNR", avg_psnr_value, epoch + 1)
        # Print evaluation indicators.
        print(f"Valid stage: {stage} Epoch[{epoch + 1:04d}] avg PSNR: {avg_psnr_value:.2f}.\n")

    return avg_psnr_value


def main() -> None:
    # Create a super-resolution experiment result folder.
    if not os.path.exists(exp_dir1):
        os.makedirs(exp_dir1)
    if not os.path.exists(exp_dir2):
        os.makedirs(exp_dir2)

    # Load the dataset.
    train_dataset = BaseDataset(train_dir, image_size, upscale_factor, "train")
    valid_dataset = BaseDataset(valid_dir, image_size, upscale_factor, "valid")
    train_dataloader = DataLoader(train_dataset, batch_size, True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, False, pin_memory=True)
    # Check whether the training progress of the last abnormal end is restored, for example, the power is
    # cut off in the middle of the training.
    if resume:
        print("Resuming...")
        if resume_p_weight != "":
            generator.load_state_dict(torch.load(resume_p_weight))
        else:
            discriminator.load_state_dict(torch.load(resume_d_weight))
            generator.load_state_dict(torch.load(resume_g_weight))

    # Initialize the evaluation indicators for the training stage of the generator model.
    best_psnr_value = 0.0
    # Train the generative network stage.
    for epoch in range(start_p_epoch, p_epochs):
        # Train each epoch for generator network.
        train_generator(train_dataloader, epoch)
        # Verify each epoch for generator network.
        psnr_value = validate(valid_dataloader, epoch, "generator")
        # Determine whether the performance of the generator network under epoch is the best.
        is_best = psnr_value > best_psnr_value
        best_psnr_value = max(psnr_value, best_psnr_value)
        # Save the weight of the generator network under epoch. If the performance of the generator network under epoch
        # is best, save a file ending with `-best.pth` in the `results` directory.
        torch.save(generator.state_dict(), os.path.join(exp_dir1, f"p_epoch{epoch + 1}.pth"))
        if is_best:
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "p-best.pth"))

    # Save the weight of the last generator network under Epoch in this stage.
    torch.save(generator.state_dict(), os.path.join(exp_dir2, "p-last.pth"))

    # Initialize the evaluation index of the adversarial network training phase.
    best_psnr_value = 0.0
    # Load the model weights with the best indicators in the previous round of training.
    generator.load_state_dict(torch.load(os.path.join(exp_dir2, "p-best.pth")))
    # Training the adversarial network stage.
    for epoch in range(start_epoch, epochs):
        # Train each epoch for adversarial network.
        train_adversarial(train_dataloader, epoch)
        # Verify each epoch for adversarial network.
        psnr_value = validate(valid_dataloader, epoch, "adversarial")
        # Determine whether the performance of the adversarial network under epoch is the best.
        is_best = psnr_value > best_psnr_value
        best_psnr_value = max(psnr_value, best_psnr_value)
        # Save the weight of the adversarial network under epoch. If the performance of the adversarial network
        # under epoch is the best, it will save two additional files ending with `-best.pth` in the `results` directory.
        torch.save(discriminator.state_dict(), os.path.join(exp_dir1, f"d_epoch{epoch + 1}.pth"))
        torch.save(generator.state_dict(), os.path.join(exp_dir1, f"g_epoch{epoch + 1}.pth"))
        if is_best:
            torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-best.pth"))
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-best.pth"))
        # Adjust the learning rate of the adversarial model.
        d_scheduler.step()
        g_scheduler.step()

    # Save the weight of the adversarial network under the last epoch in this stage.
    torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-last.pth"))
    torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-last.pth"))


if __name__ == "__main__":
    main()
