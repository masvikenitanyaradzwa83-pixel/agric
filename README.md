# AgroVision – Climate-Smart Agricultural Advisory System

**Target users**: Smallholder farmers in Gweru District, Midlands Province, Zimbabwe.

## Problem
Climate variability, limited timely information, and low productivity. Decisions on planting, crop choice and pest management are often not data-driven.

## Solution
A full-stack advisory system that:
1. Accepts farmer inputs (farm size, soil pH, input access, past yield, risk preference)
2. Recommends crops using soil pH, rainfall reliability, temperature and risk preference (PyTorch MLP + agronomic knowledge base)
3. Predicts optimal planting windows from historical rainfall onset (NASA POWER daily data for Gweru)
4. Issues pest/disease alerts based on recent temperature, humidity and crop stage
5. Visualises results in a simple, mobile-friendly web interface

## Tech stack
- **Data**: NASA POWER daily reanalysis (2015–2024) for Gweru (19.45°S, 29.82°E)
- **Models**: PyTorch `CropSuitabilityNet` (MLP + crop embedding)
- **Notebook**: Colab-ready EDA + training (`notebooks/01_AgroVision_Training_and_EDA.ipynb`)
- **Serving**: Streamlit (farmer UI) + FastAPI-ready engine
- **Best practices**: Reproducible seeds, knowledge-base soft labels when outcome labels are scarce, TorchScript export, clear climate context, explainable scores

## Project layout
```
agro_advisory/
├── data/
│   └── gweru_nasa_power_2015_2024.csv   # real NASA POWER data
├── models/
│   └── crop_suitability_net.pt
├── src/
│   ├── crop_knowledge.py      # agronomic rules + scoring
│   ├── weather_analysis.py    # onset detection, climatology
│   ├── models.py              # PyTorch net + training
│   └── advisory_engine.py     # orchestration
├── app/
│   └── streamlit_app.py       # farmer-facing UI
├── notebooks/
│   └── 01_AgroVision_Training_and_EDA.ipynb
├── requirements.txt
└── README.md
```

## Quick start

### 1. Install
```bash
cd agro_advisory
pip install -r requirements.txt
```

### 2. Run the web app
```bash
streamlit run app/streamlit_app.py
```

### 3. (Optional) Retrain on Colab
Open `notebooks/01_AgroVision_Training_and_EDA.ipynb` in Google Colab, run all cells, download the `.pt` weights and place them in `models/`.

## Example output (risk-averse + low inputs, pH 5.1)
- Top crops: Finger Millet, Sorghum, Pearl Millet / Cowpea
- Planting window derived from historical onset statistics
- Pest alerts (Fall Armyworm, Stalk Borer) when recent weather is favourable for development

## Data & model notes
- Long-term average annual rainfall at Gweru ≈ 730 mm (NASA POWER 2015–2024)
- Mean temperature ≈ 19.5 °C
- Gweru largely falls in Natural Regions III–IV (semi-intensive to semi-extensive)
- Model trained on soft labels derived from local crop optima + drought tolerance literature; can be fine-tuned with real yield feedback later

## Disclaimer
This system supports decision-making. Always cross-check with local agricultural extension officers and the latest seasonal forecasts from the Meteorological Services Department of Zimbabwe.
