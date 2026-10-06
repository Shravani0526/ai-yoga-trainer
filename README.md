# Virtual Yoga Trainer Using MoveNet

A college-project web app that watches you practise yoga through a webcam (or an uploaded video), measures your
joint angles with **Google MoveNet**, and gives **explainable, joint-level corrections** such as:

> **Right knee: 132°. Target: 90–110°. Bend your right knee slightly.**

Poses: **Tree Pose, Warrior II, Chair Pose, Triangle Pose, T Pose.**

## Honest scope (read this before your viva)

* **Pose estimation** = MoveNet (a pre-trained Google model). It is used as-is; nothing is retrained.
* **Pose classification and alignment are rule-based (geometric).** Every target range is a visible number in
  `ai/yoga/poses.py`. There is **no trained classifier and no accuracy figure** in this project – none is claimed.
* Joint angles are measured in 2-D image space from a single camera, so camera placement matters
  (front view for most poses, **side view for Chair Pose**).
* The ranges are reasonable teaching defaults, not clinical standards. Not medical advice.
* The unit tests use synthetic stick-figure skeletons. They prove the rule engine behaves as designed; they do not
  measure MoveNet's accuracy on real people.

## Core pipeline

```
Camera/Video → MoveNet → 17 keypoints → normalisation → smoothing → joint angles → pose classification
→ alignment analysis → explainable correction → scoring → stability → hold timer → session history
→ personalisation → recommendations
```

| Stage | Module |
|---|---|
| MoveNet detection (TFLite) | `ai/pose/movenet.py`, `ai/models/registry.py` |
| Keypoint normalisation (hip-centred, torso-scaled) | `ai/pose/normalization.py` |
| Smoothing (One-Euro filter) | `ai/pose/smoothing.py` |
| Angles / distances | `ai/pose/geometry.py` |
| Pose rules + classifier | `ai/yoga/poses.py`, `criteria.py`, `classifier.py` |
| Alignment | `ai/yoga/alignment.py` |
| Explainable corrections | `ai/feedback/corrections.py` |
| Scoring / symmetry / stability | `ai/yoga/scoring.py`, `symmetry.py`, `stability.py` |
| Hold timer (runs only when correct **and** stable) | `ai/yoga/hold_timer.py` |
| Session recording | `ai/yoga/session.py` |
| Calibration, adaptive difficulty, recurring errors, progress | `ai/personalization/` |
| Recommendations | `ai/feedback/recommendations.py` |
| Everything wired together (used by app **and** notebook) | `ai/pipeline.py` |

### Scoring formula

* Symmetric poses (T Pose, Chair Pose): `0.65 × alignment + 0.20 × stability + 0.15 × symmetry`
* One-sided poses (Tree, Warrior II, Triangle): `0.75 × alignment + 0.25 × stability`
* Each joint rule scores 100 inside its target range and falls linearly to 0 at its tolerance distance.

### Hold timer

Counts only while the pose is recognised, alignment ≥ the level's minimum, no key joint is badly off, and stability ≥ the
level's minimum. A 0.4 s entry delay and 0.6 s grace period ignore tracking glitches; paused time is never added.

### Personalisation

* **Calibration:** you hold your comfortable best pose; a target range is *extended* (never tightened) towards your
  measured value by at most 12° (0.30 torso lengths for distances). Stance/foot-lift targets are scaled by your leg-to-torso ratio.
  A calibration applies only to the side it was recorded for.
* **Adaptive difficulty:** Beginner / Intermediate / Advanced change range width, minimum scores and hold target.
  Promotion: last two sessions of a pose both ≥ 80 **and** hold completed. Demotion: both < 45.
* **Recurring errors:** a joint out of range for ≥ 30 % of a session, in ≥ 2 of the last 6 sessions (and ≥ half of those
  where it was measured).

## Project structure

```
ai-yoga-trainer/
├── app.py                      Streamlit app (all pages)
├── ai/
│   ├── pipeline.py             End-to-end pipeline (shared by app and notebook)
│   ├── models/                 MoveNet model registry/downloader (+ .tflite file once downloaded)
│   ├── pose/                   keypoints, MoveNet, normalisation, smoothing, geometry
│   ├── yoga/                   poses, criteria, classifier, alignment, scoring, stability, symmetry, hold timer, session
│   ├── feedback/               corrections, recommendations
│   └── personalization/        calibration, difficulty, recurring errors, progress
├── database/                   SQLite / PostgreSQL layer, schema.sql, init_db.py
├── utils/                      config, drawing, video, plotting, webcam (WebRTC), synthetic skeletons
├── notebooks/yoga_trainer_colab.ipynb
├── tests/                      unit tests (unittest/pytest compatible)
├── requirements.txt            requirements-colab.txt
├── .env.example  .gitignore  LICENSE  README.md  .streamlit/config.toml
```

## 1. Local setup

Python **3.10 – 3.12** recommended (3.11 is the safest for TensorFlow and Streamlit Cloud).

```bash
git clone https://github.com/<your-username>/ai-yoga-trainer.git
cd ai-yoga-trainer
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # optional (Windows: copy .env.example .env)
```

### MoveNet model file

The app downloads the model on first start into `ai/models/`. For a reliable demo (and for deployment) **download once and
commit it**:

```bash
python -c "from ai.models.registry import ensure_model; print(ensure_model('lightning'))"
git add ai/models/movenet_singlepose_lightning_float16.tflite
```

If the automatic download is blocked, download *MoveNet SinglePose Lightning (TFLite, float16)* from TensorFlow Hub / Kaggle
Models yourself and save it as `ai/models/movenet_singlepose_lightning_float16.tflite` (or set `MOVENET_MODEL_PATH`).
The Colab notebook (section 11) also lets you download the file.

### Database setup (SQLite)

Nothing to install – the app creates `data/yoga_trainer.db` automatically. To create/verify it manually:

```bash
python -m database.init_db
```

(`database/schema.sql` documents the tables: `users`, `sessions`, `session_errors`, `calibrations`, `user_levels`.)

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501, enter your name in the sidebar, choose a pose, (optionally) calibrate, then train.

### Test

```bash
python -m unittest discover -s tests -t . -v      # or:  pip install pytest && pytest -q
```

## 2. Google Colab setup

1. Push the project to GitHub (section 3) **or** zip the `ai-yoga-trainer` folder.
2. In Colab: **File → Upload notebook** → `notebooks/yoga_trainer_colab.ipynb` (or open it from GitHub).
3. Set `REPO_URL` in the first cell to your repository, or leave the placeholder and upload the zip when asked.
4. **Runtime → Run all** up to section 4 (self-check with synthetic skeletons – no camera needed).
5. For real analysis run **one** input cell in section 5 (upload a video / webcam snapshot / upload an image), then sections 6–9.

The notebook imports the same `ai/`, `utils/` and `database/` modules as the web app, so results are identical.

## 3. GitHub setup

```bash
git init
git add .
git commit -m "Virtual Yoga Trainer using MoveNet"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-yoga-trainer.git
git push -u origin main
```

Create the empty repository on github.com first (no README/licence – they already exist). Commit the `.tflite` model file
(about 5 MB) so deployment needs no download. `.env`, `data/*.db` and `.streamlit/secrets.toml` are git-ignored.

## 4. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**; choose your repository, branch `main`, main file path **`app.py`**.
3. Open **Advanced settings** and select **Python 3.11**.
4. (Optional but recommended) paste secrets in TOML:

   ```toml
   DATABASE_URL = "postgresql://USER:PASSWORD@HOST:5432/DBNAME"   # permanent history (see below)
   MOVENET_VARIANT = "lightning"
   # TURN_URL = "turn:your.turn.server:3478"   # only if the webcam does not connect on your network
   ```
5. Click **Deploy**. The first build installs TensorFlow (several minutes). After deployment the app runs on its own –
   **Colab does not need to stay open.**
6. Open the app URL, press **Start** on the Training page and allow camera access in the browser.

### Making history permanent (important)

Streamlit Community Cloud's file system is **ephemeral**: a SQLite file is wiped when the app restarts or is redeployed.
Options:

* **PostgreSQL (recommended for a live demo):** create a free database at Supabase or Neon, copy its connection string
  into the `DATABASE_URL` secret. Tables are created automatically on first start. (This path uses `psycopg2`; the
  SQLite path is what the unit tests cover.)
* **Backup file:** sidebar → *Backup & data* → *Download my data (JSON)*, and *Restore from backup* after a restart.
* **Local SQLite** is perfect for running on your own computer.

## 5. How to demonstrate it

1. **Home** – show the pipeline diagram and the "Method, transparency and limitations" box.
2. **Pose Selection** – pick Warrior II; open *Target ranges used for this pose*.
3. **Calibration** – hold the pose for ~5–10 s, save, show which ranges changed (and the 12° cap).
4. **Training → Live webcam** – deliberately straighten the front knee: show the correction text, the red joint on the
   skeleton and the paused hold timer; fix it and watch the timer resume.
5. Repeat for 2–3 sessions, then show **Session Results**, **Progress Dashboard**, **Recurring Errors**, **Recommendations**.
6. No camera in the exam hall? Use **Training → Demo (no camera)** or upload a video.

## Troubleshooting

| Problem | Fix |
|---|---|
| Camera never starts on the deployed app | Allow camera permission; try Chrome; some college/corporate networks block WebRTC – set a TURN server (secrets) or use the **Upload video** / **Photo snapshot** tabs. |
| "MoveNet model could not be loaded" | Place the `.tflite` file in `ai/models/` (see *MoveNet model file*). |
| Pose not recognised | Show your whole body; face the camera (Chair Pose: side view); good light; plain background. |
| Scores jump around | Use Lightning/Thunder in good light; wear fitted clothing; keep the camera still. |
| History disappeared after redeploy | Use PostgreSQL via `DATABASE_URL` or the JSON backup. |
| Slow on Streamlit Cloud | Use the Lightning model; analyse shorter videos. |

## Customising a pose

Edit `ai/yoga/poses.py`. Each rule is a `Criterion(key, label, metric, low, high, tol, weight, too_low, too_high, ...)`.
Add `signature=True` for rules that define the pose for the classifier (`gate=True` makes a rule mandatory). Sided poses are
defined once for the left side and mirrored automatically. Run the tests afterwards.

## Final run checklist

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .      # all tests pass
python -c "from ai.models.registry import ensure_model; ensure_model('lightning')"
streamlit run app.py
```

## Licence & credits

MIT licence (see `LICENSE`). MoveNet © Google, Apache-2.0, via TensorFlow Hub / Kaggle Models.
