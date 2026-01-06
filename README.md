# gan-style-transfer

This repository explores image-to-image translation using Generative Adversarial Networks (GANs),
following a single use-case comparative approach. All models address the same task: translating
images from one domain to another (Apple ↔ Orange).

---

## Task

- Apple ↔ Orange translation
- Domain A: apple images
- Domain B: orange images
- Data pairing: unpaired (where applicable)
- Image resolution: 256×256
- Dataset: huggan/apple2orange (~1,000 images per domain)

---

## Models

- DCGAN
- UNet-GAN / Pix2Pix-style
- CycleGAN
- StyleGAN (fine-tuning)

---

## Evaluation

- FID
- SSIM
- Visual inspection
- Content preservation
- Style consistency
- Training stability
- Computational cost

---

## Team Members

- Tisca Laurentiu — DCGAN
- Soltuz Rares — UNet-GAN / Pix2Pix-style
- Priboi Andrei — CycleGAN
- Ignat Dan — StyleGAN

---

## Notes

Complex architectures rely on transfer learning or fine-tuning due to computational constraints.
The focus is on pipeline correctness, observed behaviors, and comparative analysis rather than
state-of-the-art performance.

---

## References

Goodfellow et al., *Generative Adversarial Nets*, 2014  
Radford et al., *Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN)*, 2015  
Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)*  
Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)*  
Karras et al., *A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)*  
Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)*  
Wang et al., *Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)*