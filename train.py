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

import torch.utils.data

from config import *
from dataset import BaseDataset


def main() -> None:
    discriminator.train()
    generator.train()

    # Load dataset.
    dataset = BaseDataset(dataroot, image_size, scale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, True, pin_memory=True)
    # Resuming training.
    if resume:
        print("Resuming...")
        discriminator.load_state_dict(torch.load(resume_d_weight))
        generator.load_state_dict(torch.load(resume_g_weight))

    num_batches = len(dataloader)
    # Train PSNR-Oral.
    for epoch in range(start_p_epoch, p_epochs):
        for index, data in enumerate(dataloader, 1):
            ##############################################
            # (0) Update G network: min L1 Loss(output, hr)
            ##############################################
            generator.zero_grad()
            lr = data[0].to(device)
            hr = data[1].to(device)

            sr = generator(lr)
            loss = pixel_criterion(sr, hr)
            loss.backward()
            p_optimizer.step()
            # Print Loss.
            print(f"Epoch[{epoch:05d}/{p_epochs:05d}]({index:05d}/{num_batches:05d}) Loss: {loss.item():.4f}.")
            batches = index + epoch * num_batches + 1
            writer.add_scalar("PSNR/Loss", loss.item(), batches)
        # Save checkpoint model.
        torch.save(generator.state_dict(), os.path.join("samples", f"P_epoch{epoch}.pth"))
    # Save final model.
    torch.save(generator.state_dict(), os.path.join("results", "P-last.pth"))

    # Train GAN-Oral.
    for epoch in range(start_g_epoch, g_epochs):
        for index, data in enumerate(dataloader, 1):
            discriminator.zero_grad()
            lr = data[0].to(device)
            hr = data[1].to(device)
            label_size = lr.size(0)
            # Real sample label is 1, fake sample label is 0.
            real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, device=device)
            fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, device=device)
            ##############################################
            # (1) Update D network: E(HR)[log(D(HR))] + E(SR)[log(1 - D(G(SR))]
            ##############################################
            # Train real sample with discriminator.
            hr_output = discriminator(hr)
            d_loss_hr = adversarial_criterion(hr_output, real_label)
            d_loss_hr.backward()
            d_hr = hr_output.mean().item()

            # Train fake sample with discriminator.
            sr = generator(lr)
            sr_output = discriminator(sr.detach())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
            d_loss_sr.backward()
            d_sr1 = sr_output.mean().item()
            d_loss = d_loss_hr + d_loss_sr
            d_optimizer.step()
            ##############################################
            # (2) Update G network: E(SR)[log(1 - D(G(SR))]
            ##############################################
            generator.zero_grad()

            # Train fake sample with generator.
            sr = generator(lr)
            sr_output = discriminator(sr)
            pixel_loss       = pixel_weight       * pixel_criterion(sr, hr.detach())
            perceptual_loss  = perceptual_weight  * perceptual_criterion(sr, hr.detach())
            adversarial_loss = adversarial_weight * adversarial_criterion(sr_output, real_label)
            d_sr2 = sr_output.mean().item()
            g_loss = pixel_loss + perceptual_loss + adversarial_loss
            g_loss.backward()
            g_optimizer.step()
            # Print Loss.
            print(f"Epoch[{epoch:05d}/{g_epochs:05d}]({index:05d}/{num_batches:05d}) "
                  f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}"
                  f"D(HR): {d_hr} D(SR1)/D(SR2): {d_sr1}/{d_sr2}.")
            batches = index + epoch * num_batches + 1
            writer.add_scalar("GAN/D_Loss", d_loss.item(), batches)
            writer.add_scalar("GAN/G_Loss", g_loss.item(), batches)
        # Adjust learning rate.
        d_scheduler.step()
        g_scheduler.step()
        # Save checkpoint model.
        torch.save(discriminator.state_dict(), os.path.join("samples", f"D_epoch{epoch}.pth"))
        torch.save(generator.state_dict(),     os.path.join("samples", f"G_epoch{epoch}.pth"))
    # Save final model.
    torch.save(discriminator.state_dict(), os.path.join("results", "D-last.pth"))
    torch.save(generator.state_dict(),     os.path.join("results", "G-last.pth"))


if __name__ == "__main__":
    main()
