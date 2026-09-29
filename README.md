# AI-Based Routing Congestion Prediction — Project Skeleton

## Libraries used

| Purpose                          | Library                              |
|-----------------------------------|---------------------------------------|
| Chip placement & routing          | OpenROAD (via Docker, controlled with Tcl scripts) |
| Numeric arrays / grid building    | NumPy                                |
| Reading/writing tabular reports   | Pandas                               |
| Deep learning model (U-Net)       | PyTorch                              |
| Evaluation (SSIM)                 | scikit-image                         |
| Plotting heatmaps                 | Matplotlib                           |
| Progress bars                     | tqdm                                 |

## Pipeline (matches your 6-step plan)

1. `tcl_scripts/extract_placement.tcl` — runs INSIDE the OpenROAD docker.
   Loads a design, runs placement, dumps every cell's position and every
   pin's position to CSV files.

2. `tcl_scripts/global_route_congestion.tcl` — runs INSIDE OpenROAD.
   Runs global routing on the placed design and dumps a congestion report
   (per-gcell overflow) to CSV. This is your "ground truth".

3. `src/feature_engineering.py` — runs OUTSIDE docker, plain Python.
   Reads the placement CSVs, bins everything into a grid (e.g. 256x256),
   and computes 3 channels: cell density, pin density, RUDY.
   Saves as a `.npy` file (input X).

4. `src/heatmap_generation.py` — plain Python.
   Reads the congestion CSV, bins it onto the SAME grid, saves as `.npy`
   (target Y).

5. `src/dataset.py` — PyTorch Dataset that loads (X, Y) pairs.

6. `src/model.py` — U-Net architecture definition.

7. `src/train.py` — training loop (80/20 split, saves checkpoints).

8. `src/evaluate.py` — loads a trained model, predicts on the test set,
   computes MSE and SSIM, and plots predicted vs real heatmaps.

## How you'd actually run it

```bash
# Inside the OpenROAD docker container:
openroad tcl_scripts/extract_placement.tcl
openroad tcl_scripts/global_route_congestion.tcl

# Outside docker, in your normal Python env:
pip install -r requirements.txt
python src/feature_engineering.py
python src/heatmap_generation.py
python src/train.py
python src/evaluate.py
```

Note: the Tcl scripts assume you already have a design (e.g. OpenROAD's
built-in `gcd` or `ibex` example, or an ISPD 2011 benchmark) loaded the
same way OpenROAD's own flow scripts do. You will need to adjust file
paths (`.def`, `.lef`) to match whichever design you use.
