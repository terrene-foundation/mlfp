# M5 Instructor Colab Set

Pre-filled Colab notebooks for Module 5 — these contain the full solution
code (no `____` blanks) so you can open them in Google Colab and run
end-to-end to verify:
- `DLDiagnostics` imports and fires
- `diag.report()` prints the Prescription Pad
- `diag.plot_training_dashboard().show()` renders the 4-panel dashboard
- All training loops complete
- Grad-CAM, attention heatmaps, etc. produce outputs

## Running in Colab

1. Open any notebook (e.g. `ex_1/01_standard_ae.ipynb`) in
   [Google Colab](https://colab.research.google.com)

2. **Cell 0** — edit `FORK_URL`:
   ```python
   FORK_URL = "https://github.com/pcml-run26/pcml-run26-2601.git"
   ```
   (As instructor you can point at the template repo directly — students
   point at their fork.)

3. Run cell 0 — clones the repo, installs deps, adds to `sys.path`

4. Run all cells — every cell is filled in with working solution code.
   The diagnostic checkpoint fires near the end of each notebook and
   prints a real Prescription Pad based on the actual training run.

## What students get (the differences)

Student notebooks (`../colab/ex_N/`) have the same structure but with
`____` blanks in the model definitions, training loops, and visualisations
for students to fill in. The diagnostic block at the end is unchanged —
students see captured reference output inline and compare to their own
runs.

## Not distributed to students

This directory is instructor-only. Do not share with students (it contains
full solutions). The student repo sync script excludes `colab-instructor/`.
