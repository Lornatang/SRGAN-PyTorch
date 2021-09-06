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
    # The purpose of training the generator is to minimize the pixel difference
    # between super-resolution images and high-resolution images.
    batches = len(train_dataloader)

    # Start training mode.
    generator.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        # 0. Copy the data to the designated device.
        lr = lr.to(device)
        hr = hr.to(device)
        # 1. Set the gradient of the generated model to 0.
        generator.zero_grad()
        # 2. Calculate the difference between the super-resolution image and the high-resolution image at the pixel level.
        sr = generator(lr)
        pixel_loss = pixel_criterion(sr, hr)
        # 3. Update the weights of the generative model.
        pixel_loss.backward()
        p_optimizer.step()

        # Write the loss value in the process of training and generating the network into Tensorboard.
        iters = index + epoch * batches + 1
        writer.add_scalar("Train_Generator/Loss", pixel_loss.item(), iters)
        # Output the loss function every one hundred iterations.
        if (index + 1) % 100 == 0 or (index + 1) == batches:
            print(f"Train stage: generator "
                  f"Epoch[{epoch + 1:04d}/{p_epochs:04d}]({index + 1:05d}/{batches:05d}) "
                  f"Loss: {pixel_loss.item():.6f}.")


def train_adversarial(train_dataloader, epoch) -> None:
    # The purpose of training the generator is to minimize the pixel difference between super-resolution images and high-resolution images.
    batches = len(train_dataloader)

    # Start training mode.
    discriminator.train()
    generator.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        # 0. Copy the data to the designated device.
        lr = lr.to(device)
        hr = hr.to(device)

        label_size = lr.size(0)
        # 1. Tagging. Set the real sample label to 1, and the false sample label to 0.
        real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, device=device)
        fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, device=device)

        # 2. Set the discriminator gradient to 0.
        discriminator.zero_grad()
        # 3. Calculate the loss of the discriminator on the high-resolution image.
        output = discriminator(hr)
        d_loss_hr = adversarial_criterion(output, real_label)
        d_loss_hr.backward()
        d_hr = output.mean().item()
        # 4. Calculate the loss of the discriminator on the super-resolution image.
        sr = generator(lr)
        output = discriminator(sr.detach())
        d_loss_sr = adversarial_criterion(output, fake_label)
        d_loss_sr.backward()
        d_sr1 = output.mean().item()
        # 5. Update the discriminator weights.
        d_loss = d_loss_hr + d_loss_sr
        d_optimizer.step()

        # 6. Set the generator gradient to 0.
        generator.zero_grad()
        # 7. Calculate the loss of the generator on the super-resolution image.
        output = discriminator(sr)
        # Weighted loss. Pixel loss + perceptual loss + confrontation loss.
        pixel_loss = pixel_weight * pixel_criterion(sr, hr.detach())
        perceptual_loss = perceptual_weight * perceptual_criterion(sr, hr.detach())
        adversarial_loss = adversarial_weight * adversarial_criterion(output, real_label)
        # 8. Update generator weights.
        g_loss = pixel_loss + perceptual_loss + adversarial_loss
        g_loss.backward()
        g_optimizer.step()
        d_sr2 = output.mean().item()

        # Write the loss value in the process of training the adversarial network into Tensorboard.
        iters = index + epoch * batches + 1
        writer.add_scalar("Train_Adversarial/D_Loss", d_loss.item(), iters)
        writer.add_scalar("Train_Adversarial/G_Loss", g_loss.item(), iters)
        writer.add_scalar("Train_Adversarial/D_HR", d_hr, iters)
        writer.add_scalar("Train_Adversarial/D_SR1", d_sr1, iters)
        writer.add_scalar("Train_Adversarial/D_SR2", d_sr2, iters)
        # Output the loss function every one hundred iterations.
        if (index + 1) % 100 == 0 or (index + 1) == batches:
            print(f"Train stage: adversarial "
                  f"Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                  f"D Loss: {d_loss.item():.6f} G Loss: {g_loss.item():.6f} "
                  f"D(HR): {d_hr:.6f} D(SR1)/D(SR2): {d_sr1:.6f}/{d_sr2:.6f}.")


def validate(valid_dataloader, epoch, stage) -> float:
    # Verify the performance of the model after each Epoch.
    batches = len(valid_dataloader)
    total_psnr_value = 0.0

    # Start verification mode.
    generator.eval()

    with torch.no_grad():
        for index, (lr, hr) in enumerate(valid_dataloader):
            # Copy the data to the specified device.
            lr = lr.to(device)
            hr = hr.to(device)
            # Calculate the loss.
            sr = generator(lr)
            # Calculate the PSNR indicator.
            mse_loss = ((sr - hr) ** 2).data.mean()
            psnr_value = 10 * torch.log10(1 / mse_loss).item()

            # Output indicators every one hundred iterations.
            if (index + 1) % 100 == 0 or (index + 1) == batches:
                print(f"Test stage: {stage} "
                      f"Epoch[{epoch + 1:04d}]({index + 1:04d}/{batches:04d}) "
                      f"PSNR: {psnr_value:.2f}.")
            total_psnr_value += psnr_value

        avg_psnr_value = total_psnr_value / batches
        # Write the value of each round of verification indicators into Tensorboard.
        if stage == "generator":
            writer.add_scalar("Val_Generator/PSNR", avg_psnr_value, epoch + 1)
        elif stage == "adversarial":
            writer.add_scalar("Val_Adversarial/PSNR", avg_psnr_value, epoch + 1)
        # Print evaluation indicators.
        print(f"Test stage: {stage} "
              f"Epoch[{epoch + 1:04d}] avg PSNR: {avg_psnr_value:.2f}.")

    return avg_psnr_value


def main() -> None:
    # Create a folder of super-resolution experiment results.
    if not os.path.exists(exp_dir1):
        os.makedirs(exp_dir1)
    if not os.path.exists(exp_dir2):
        os.makedirs(exp_dir2)

    # Load the data set.
    train_dataset = BaseDataset(train_dir, image_size, upscale_factor, "train")
    valid_dataset = BaseDataset(valid_dir, image_size, upscale_factor, "valid")
    train_dataloader = DataLoader(train_dataset, batch_size, True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, False, pin_memory=True)
    # Check whether the training progress of the last abnormal end is restored, for example,
    # the power is cut off in the middle of the training.
    if resume:
        print("Resuming...")
        if resume_p_weight != "":
            generator.load_state_dict(torch.load(resume_p_weight))
        else:
            discriminator.load_state_dict(torch.load(resume_d_weight))
            generator.load_state_dict(torch.load(resume_g_weight))

    # Initialize the evaluation index of the generator training phase
    best_psnr_value = 0.0
    # Train the generator stage.
    for epoch in range(start_p_epoch, p_epochs):
        # Train the generator model under each Epoch.
        train_generator(train_dataloader, epoch)
        # Verify the model performance after each Epoch.
        psnr_value = validate(valid_dataloader, epoch, "generator")
        # Evaluate whether the performance of the current node model is the highest indicator.
        is_best = psnr_value > best_psnr_value
        best_psnr_value = max(psnr_value, best_psnr_value)
        # Save the node model. If the current node model has the best performance,
        # a model file ending with `-best` will also be saved.
        torch.save(generator.state_dict(), os.path.join(exp_dir1, f"p_epoch{epoch + 1}.pth"))
        if is_best:
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "p-best.pth"))

    # Save the training model for the last iteration of the training generator phase.
    torch.save(generator.state_dict(), os.path.join(exp_dir2, "p-last.pth"))

    # Initialize the evaluation index of the adversarial network training stage.
    best_psnr_value = 0.0
    # Load the model with the best performance in the training phase of the generator.
    generator.load_state_dict(torch.load(os.path.join(exp_dir2, "p-best.pth")))
    # Training against the network stage.
    for epoch in range(start_epoch, epochs):
        # Train the generator model and discriminator model under each Epoch.
        train_adversarial(train_dataloader, epoch)
        # Verify the model performance after each Epoch.
        psnr_value = validate(valid_dataloader, epoch, "adversarial")
        # Evaluate whether the performance of the current node model is the highest indicator.
        is_best = psnr_value > best_psnr_value
        best_psnr_value = max(psnr_value, best_psnr_value)
        # Save the node model. If the current node model has the best performance, a model file ending with `-best` will also be saved.
        torch.save(discriminator.state_dict(), os.path.join(exp_dir1, f"d_epoch{epoch + 1}.pth"))
        torch.save(generator.state_dict(), os.path.join(exp_dir1, f"g_epoch{epoch + 1}.pth"))
        if is_best:
            torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-best.pth"))
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-best.pth"))

    # Save the training model for the last iteration of the training adversarial network phase.
    torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-last.pth"))
    torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-last.pth"))
    # Adjust the learning rate.
    d_scheduler.step()
    g_scheduler.step()


if __name__ == "__main__":
    main()
