# iflens - Generative AI Data Augmentation Tool
![Static Badge](https://img.shields.io/badge/Gemini-OK-green?logo=googlegemini) ![Static Badge](https://img.shields.io/badge/Groq-OK-green?logo=groq) ![Static Badge](https://img.shields.io/badge/OpenAI-WIP-yellow?logo=openai) ![Static Badge](https://img.shields.io/badge/Claude-TODO-black?logo=claude)

## Installation

### Clone the repository
```bash
git clone https://github.com/Starblessed/iflens.git .
```

### Install the dependencies

uv (recommended)
```bash
uv sync
```

python
```bash
python -m venv .venv

.venv\Scripts\activate # Windows
.venv\bin\activate # Linux

pip install .
```

## Usage

### Condition Configuration

Edit the `conditions-example.yaml` file with the desired conditions for image classification.

```yaml
people:
  empty: No people visible in the scene.
  some: Some people visible in the scene.
  crowded: Lively scene crowded with lots of people.

time_of_day:
  dawn: Early daylight with low ambient illumination.
  morning: Morning daylight with moderate illumination.
  noon: Bright overhead daylight with strong illumination.
  afternoon: Afternoon daylight with warm angled illumination.
  dusk: Fading daylight with low ambient illumination.
  night: Nighttime with minimal natural illumination.

precipitation:
  clean: No precipitation.
  light_rain: Light rainfall with minor visibility reduction.
  heavy_rain: Heavy rainfall with significant visibility reduction.
  stormfront: Severe storm with intense rainfall and turbulent conditions.
  snow: Light snowfall with minor visibility reduction.
  heavy_snow: Heavy snowfall with significant visibility reduction.
  blizzard: Severe snowfall with strong winds and low visibility.

planet:
  earth: Standard picture taken on planet Earth.
  mars: Picture taken on planet Mars.
```

### API Keys

Make a copy of the `.env.example` file, rename it to `.env`, and fill in the requested API keys:

```ini
GEMINI_API_KEY=YOUR_KEY_HERE
GROQ_API_KEY=YOUR_KEY_HERE
```

### Pipeline Configuration

Modify the generation variables in the `iflens/main.py` script:

```python
NUMBER_OF_SAMPLE_IMAGES: int = 1 # Should be equal to or smaller than the dataset size

DATASET_PATH: str = os.path.join("examples", "street")
CONDITIONS_YAML_FILE: str = "conditions-example.yaml"

KEEP_CLASSES: list[str] = ["planet"] # Prevents the tool from varying specific condition classes

GENERATION_MODEL: str = "gemini-3.1-flash-lite-image"
```

### Running

Open a console in the project root directory and run:

uv (recommended)
```bash
uv run python -m iflens.main
```

python
```bash
.venv\Scripts\activate # Windows
.venv\bin\activate # Linux

python -m iflens.main
```

---
### Attribution

- [Car Picture 1](examples\cars\domaxi198-shelby-3821712.jpg): Image by <a href="https://pixabay.com/pt/users/domaxi198-10651890/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=3821712">domaxi198</a> via <a href="https://pixabay.com/pt//?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=3821712">Pixabay</a>
- [Car Picture 2](examples\cars\mrefraim1-car-2667246.jpg): Image by <a href="https://pixabay.com/pt/users/mrefraim1-6230354/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2667246">mrefraim1</a> via <a href="https://pixabay.com/pt//?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2667246">Pixabay</a>
- [Car Picture 3](examples\cars\mibro-race-car-7624025.jpg): Image by <a href="https://pixabay.com/pt/users/mibro-8455312/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=7624025">mibro</a> via <a href="https://pixabay.com/pt//?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=7624025">Pixabay</a>

## Reproducible runs

Every run now needs a master seed, and each output image gets a `AUG_<name>.json` record
(conditions before/after, seeds, full prompt, model requested/returned, response id, timestamps,
SHA-256 of input/output and of the conditions file):

```bash
python -m iflens.main --dataset path/to/images --conditions conditions-citra.yaml \
  --n 50 --seed 42 --keep
```

Notes: sampling, classification (temperature 0, Groq `seed`) and condition variation are seeded.
The image-editing call itself is **not** guaranteed deterministic (no seed is passed to the
generation model), so archive the outputs; the record is what makes them auditable.
`conditions-citra.yaml` is a restricted set for the Brazilian coast; `conditions-citra-risk.yaml`
holds optional values (heavy rain, dense fog) that may invalidate bounding boxes.
