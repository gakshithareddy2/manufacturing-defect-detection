# Manufacturing Defect Detection Dashboard

PBL dashboard for the Manufacturing Defect Detection project.

## What it does

- Upload a steel-surface image.
- Run the fine-tuned ResNet50 model.
- Display the predicted defect.
- Display confidence.
- Display probabilities for all six NEU defect classes.

Classes:
1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

## Required model

Put the trained file below in this same folder:

`final_manufacturing_defect_resnet50.keras`

This is the exact model filename used by the Colab notebook.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard resizes the image to 128x128 RGB before inference. The saved model already contains the ResNet50 preprocessing layer, matching the notebook's inference workflow.
