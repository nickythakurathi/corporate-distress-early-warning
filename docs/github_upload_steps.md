# GitHub Upload Steps

## Best repository name
Use one of these:
- `corporate-distress-early-warning`
- `early-warning-corporate-distress`
- `corporate-distress-prediction`

The first option is the strongest.

## Before uploading
Make sure this folder contains:
- `README.md`
- `requirements.txt`
- `src/`
- `presentations/`
- `notebooks/original_export/`
- `figures/` with 2 to 4 clean PNG files if available

## Recommended visibility
- **Private** if only professors should review it
- **Public** if you also want to use it as a portfolio project

## Easiest upload method on GitHub website
1. Go to GitHub
2. Click **New repository**
3. Repository name: `corporate-distress-early-warning`
4. Add description:
   - `Forward-looking corporate distress prediction using accounting ratios, CRSP market data, and machine learning.`
5. Choose public or private
6. Click **Create repository**
7. Click **uploading an existing file**
8. Drag the **contents** of this folder into GitHub
9. Commit with message:
   - `Initial research repository upload`

## After first upload
Open the repo and make sure:
- the README renders correctly,
- the slide deck opens,
- the `src/` folder is visible,
- the original notebook export is preserved.

## Best images to add to README
Export PNGs and put them in `figures/`:
- `project_workflow.png`
- `model_comparison_auc.png`
- `xgb_feature_importance.png`
- `watchlist_2024_summary.png`

## What not to upload
Do not upload:
- very large raw datasets if they are proprietary or too big,
- duplicate notebook checkpoints,
- random temporary files,
- messy local absolute paths.

## Best commit message sequence
If you upload in stages:
1. `Add project structure and README`
2. `Add cleaned model scripts`
3. `Add presentation materials and notebook export`
4. `Add figures and final documentation`
