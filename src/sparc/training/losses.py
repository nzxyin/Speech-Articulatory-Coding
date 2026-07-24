# Standard HiFi-GAN least-squares GAN losses (Kong et al. 2020), reimplemented
# since no discriminator/loss code exists elsewhere in this codebase. Loss
# weights (GAN=1, mel=45, feature-matching=2) are applied externally by the
# training module (lightning_module.py) to match the SPARC paper's Appendix
# B.6 exactly, rather than being baked into these functions.

import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0.0
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_adv_loss(disc_outputs):
    loss = 0.0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses
