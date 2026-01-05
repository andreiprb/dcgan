# gan-style-transfer

This repository explores artistic style transfer using Generative Adversarial Networks (GANs),
following a single use-case comparative approach. All models address the same task: translating
real-world photographs into a specific artistic style (e.g., Monet).

---

## Task

- Photo → Artistic Style transfer
- Content domain: real photographs
- Style domain: artistic images
- Data pairing: unpaired (where applicable)
- Image resolution: 256×256

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
Ledig et al., *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN)*  
Karras et al., *A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)*  
Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)*  
Wang et al., *Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)*
