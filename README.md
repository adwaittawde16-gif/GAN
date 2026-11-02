[README.md](https://github.com/user-attachments/files/34284190/README.md)
# Sarpagandha-GAN: Synthetic Non-Ideal Dataset Generation for Medicinal Plant Parts

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20T4%20GPU-green.svg)](https://cloud.google.com/gpu)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)

An end-to-end Generative Adversarial Network (**cGAN / Pix2Pix**) pipeline to synthesize non-ideal, field-ready datasets of **Sarpagandha (*Rauvolfia serpentina*)** under diverse illumination ranges (**Bright Sun Glare, Canopy Twilight, Night Mobile Flash**) and realistic harvest backgrounds (**Farm Soil, Jute Sacks, Wooden Boards**) annotated with active pharmaceutical phytochemicals (*Reserpine, Ajmaline*).

---

## 📌 Problem Overview & Motivation

Most computer vision models for botanical identification fail when deployed in agricultural and forest environments because training datasets consist of "ideal" studio photographs: clean backgrounds, perfect diffuse lighting, and isolated specimens.

In real-world ethnobotanical and field collection settings:
* **Lighting varies wildly**: Harsh specular solar glare, dense canopy shadow, or mobile camera flash at night with high-ISO sensor grain.
* **Backgrounds are cluttered**: Freshly harvested plant parts sit on loose muddy soil, coarse burlap gunny sacks, or rustic wooden sorting tables.
* **High Morphological Stakes**: Hallucinating petal margins or leaf venation destroys medicinal classification accuracy.

This project uses an adversarial **Conditional Relighting & Harmonization GAN** to synthetically generate these non-ideal environmental conditions while preserving 100% of the authentic botanical anatomy.

---

## 🌿 Botanical & Pharmacological Profile

Target Species: ***Rauvolfia serpentina* (L.) Benth. ex Kurz** (Family: *Apocynaceae*)  
Common Names: *Sarpagandha, Indian Snakeroot, Chandrika, Devil Pepper*

| Plant Part | Primary Active Alkaloids | Clinical & Pharmacological Utility | Visual Morphological Features |
| :--- | :--- | :--- | :--- |
| **Roots / Rhizomes** | **Reserpine** (0.15–0.2%), **Ajmaline**, **Serpentine**, Deserpidine | Potent antihypertensive (inhibits VMAT2); neuropsychiatric sedative (insomnia, schizophrenia, anxiety); anti-arrhythmic | Cylindrical, tortuous, curved with longitudinal fissures; grayish-yellow to light brown |
| **Leaves** | Trace indole alkaloids, tannins, flavonoids (kaempferol) | Traditional ophthalmic remedy for corneal opacity; juice used for wound disinfection and fever reduction | Simple, whorled phyllotaxy (3–4 at nodes), elliptic-lanceolate, glossy dark green dorsal, pale ventral |
| **Flowers** | Minor ajmaline fractions, phenolic glycosides | Traditional febrifuge; cooling eye drops; folk formulations for nervous agitation | Dense cymes; white or pinkish-white corolla with bright red/purple calyx tube |
| **Fruits / Drupes** | Seed storage lipids, minor indole derivatives | Seed propagation; topical application for colic and skin eruptions | Small globose drupes; green when unripe, transitions to shiny purplish-black upon ripening |

---

## 🔬 Architecture & Methodology

```
┌─────────────────────────┐
│ Real Plant Part (Cutout)│ ──┐
└─────────────────────────┘   │
┌─────────────────────────┐   │     ┌────────────────────────────────────┐
│ Target Background Plate │ ──┼───► │ U-Net Generator with Skip Conns (G)│ ──► [Synthesized Output Image]
└─────────────────────────┘   │     └─────────────────┬──────────────────┘
┌─────────────────────────┐   │                       │
│ Condition (0, 1, 2)     │ ──┘                       ▼
└─────────────────────────┘         ┌────────────────────────────────────┐
                                    │    70x70 Markovian PatchGAN (D)    │
                                    └────────────────────────────────────┘
```

### Why Traditional DCGAN Failed (The Research Breakthrough):
* **Noise-to-Image DCGAN ($z \to \text{Image}$)** was initially tested. With small botanical datasets ($\sim 50$–$200$ images), standard DCGAN suffered from **mode collapse** (identical copies), **checkerboard border artifacts**, and eventual **gradient saturation** (crashing into solid yellow screens).
* **The Solution**: We transitioned to a **Conditional Image-to-Image GAN (cGAN)**:
  1. **U-Net Generator with Skip Connections**: Encodes the source image while skip connections carry high-frequency spatial features (veins, petal contours) directly from the encoder to the decoder. The network only learns *how light and surfaces interact*, guaranteeing zero anatomical hallucinations.
  2. **70×70 Markovian PatchGAN Discriminator**: Judges local $70 \times 70$ pixel patches rather than the entire image as a single scalar, penalizing blurriness and enforcing sharp micro-textures.

### Objective Loss Formulation:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cGAN}}(G, D) + \lambda \mathcal{L}_{L1}(G)$$

$$\mathcal{L}_{\text{cGAN}}(G, D) = \mathbb{E}_{x, y} \left[ \log D(x, y) \right] + \mathbb{E}_{x, c} \left[ \log \left(1 - D(x, G(x, c))\right) \right]$$

$$\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, c} \left[ \| y - G(x, c) \|_1 \right] \quad (\text{with } \lambda = 100)$$

---

## 📂 Repository Structure

```
├── notebooks/
│   └── sarpagandha_synthetic_dataset.ipynb   # Complete Google Colab / T4 GPU training notebook
├── src/
│   ├── models/
│   │   ├── unet_generator.py                 # Multi-scale U-Net Generator with skip connections
│   │   └── patchgan_discriminator.py         # 70x70 Markovian PatchGAN Discriminator
│   ├── data/
│   │   ├── inaturalist_collector.py          # Automated scientific API data scraper
│   │   └── background_downloader.py          # Real agricultural texture pipeline
│   └── engines/
│       └── lighting_physics.py               # Photometric physics simulation functions
├── data/
│   └── medicinal_metadata.csv                # Tabular dataset mapping image IDs to alkaloids & uses
├── checkpoints/                              # Saved PyTorch model weights (.pth)
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### Prerequisites:
* Python 3.10+
* NVIDIA GPU (Google Colab T4 runtime recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sarpagandha-gan.git
cd sarpagandha-gan
```

### 2. Install Dependencies
```bash
pip install torch torchvision numpy opencv-python pillow matplotlib pandas requests
```

### 3. Run in Google Colab
Open [`notebooks/sarpagandha_synthetic_dataset.ipynb`](notebooks/sarpagandha_synthetic_dataset.ipynb) in Google Colab:
1. Set runtime to **T4 GPU** (**Runtime → Change runtime type → T4 GPU**).
2. Run cells sequentially to:
   * Auto-fetch peer-reviewed *Rauvolfia serpentina* specimens from iNaturalist.
   * Train the U-Net Generator and PatchGAN Discriminator.
   * Synthesize the multi-lighting, multi-background dataset.
   * Export the packaged dataset with `medicinal_metadata.csv`.

---

## 📊 Dataset Schema (`medicinal_metadata.csv`)

Every synthetic image is paired with structured pharmacological metadata:

| Field | Type | Description |
| :--- | :--- | :--- |
| `image_id` | `string` | Unique filename (e.g., `sarpagandha_flower_002_bright.jpg`) |
| `plant_part` | `string` | Botanical organ (`root`, `leaf`, `flower`, `fruit`) |
| `lighting_condition` | `string` | Simulated illumination (`bright`, `dim`, `dark`) |
| `background_type` | `string` | Environment texture (`farm_soil`, `jute_sack`, `wooden_board`) |
| `key_alkaloids` | `string` | Major active phytochemicals (*Reserpine, Ajmaline, Serpentine*) |
| `primary_medicinal_usage` | `string` | Clinical indications and Ayurvedic pharmacological value |

---

## 🛡️ License

This project is licensed under the [Apache License 2.0](LICENSE).  
Specimen photographs sourced via the [iNaturalist Open API](https://www.inaturalist.org/) under Creative Commons licenses (CC-BY, CC-BY-NC, CC0).

---

## 📖 Citation

If you use this dataset generation pipeline or architecture in your academic work, please cite:

```bibtex
@misc{sarpagandha_gan_2026,
  author = {Your Name / Team},
  title = {Synthetic Non-Ideal Dataset Generation for Medicinal Plant Parts using Conditional Relighting GANs},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/your-username/sarpagandha-gan}}
}
```
