# Here's where we will put our models

`solvers/` contains the models copied from the competition repo

## Setup (once)

Clone this repo next to your `2026-competition` clone:

## Making a new model

1. Get everyone's latest work:
   ```bash
   cd <this-repo>
   git pull
   ```
2. Copy a baseline or an existing model to a **new file** in `models/`:
   ```bash
   cp solvers/mean_ridge.py models/<your_file>.py
   ```
3. In the new file, set a **unique** name e.g.
   `name = "js_regression_model_3"`
4. Make your changes.
5. Commit the change, then pull, then push:
   ```bash
   cd ../<this-repo>
   git add models/<your_file>.py
   git commit -m "<what changed>, W-bMAE <score>"
   git pull
   git push
   ```
