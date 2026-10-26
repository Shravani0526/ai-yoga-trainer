"""Virtual Yoga Trainer Using MoveNet - Streamlit application.

Run locally:   streamlit run app.py
Pipeline:      Camera/Video -> MoveNet -> 17 keypoints -> normalisation -> smoothing -> joint angles
               -> pose classification -> alignment -> explainable correction -> scoring -> stability
               -> hold timer -> session history -> personalisation -> recommendations
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Virtual Yoga Trainer", page_icon="\U0001F9D8", layout="wide",
                   initial_sidebar_state="expanded")

from ai.feedback.corrections import format_range, format_value  # noqa: E402
from ai.feedback.impact import DISCLAIMER as IMPACT_DISCLAIMER, impact_for, session_impacts  # noqa: E402
from ai.feedback.recommendations import build_recommendations, session_feedback  # noqa: E402
from ai.models.registry import ensure_model, model_path  # noqa: E402
from ai.personalization.calibration import apply_calibration, describe_adaptation  # noqa: E402
from ai.personalization.difficulty import LEVEL_ORDER, LEVELS, apply_difficulty, recommend_level  # noqa: E402
from ai.personalization.progress import overall_summary, per_pose_summary, prepare_sessions, weekly_summary  # noqa: E402
from ai.personalization.recurring import describe_direction, detect_recurring_errors  # noqa: E402
from ai.pipeline import YogaPipeline  # noqa: E402
from ai.pose.movenet import MoveNetDetector  # noqa: E402
from ai.yoga.poses import criteria_for, get_pose, list_poses  # noqa: E402
from ai.yoga.scoring import grade_for  # noqa: E402
from database import get_database  # noqa: E402
from utils.config import (APP_SUBTITLE, APP_TITLE, get_ice_servers, get_setting, load_env_file,  # noqa: E402
                          push_streamlit_secrets_to_env)
from utils.drawing import annotate_frame  # noqa: E402
from utils.reference import reference_caption, render_reference  # noqa: E402
from utils.plotting import (hold_time_figure, joint_error_figure, progress_figure, recurring_figure,  # noqa: E402
                            timeline_figure)
from utils.synthetic import make_pose, to_pixels  # noqa: E402
from utils.video import bgr_to_rgb, decode_image, iter_video_frames, probe_video  # noqa: E402
from utils.webcam import WEBRTC_AVAILABLE, YogaProcessor  # noqa: E402

if WEBRTC_AVAILABLE:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

load_env_file()
push_streamlit_secrets_to_env()

PAGES = ["Home", "Pose Selection", "Calibration", "Training", "Session Results",
         "Progress Dashboard", "Recurring Errors", "Recommendations"]

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg, #264653 0%, #2a9d8f 100%); color: white;
       padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: .3rem 0 0 0; opacity: .92;}
.card {border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: .9rem 1.1rem; height: 100%;}
.card h4 {margin: 0 0 .35rem 0;}
.corr {border-left: 4px solid #e76f51; background: rgba(231,111,81,.09); padding: .5rem .8rem;
       border-radius: 6px; margin-bottom: .45rem; font-size: .95rem;}
.good {border-left: 4px solid #2a9d8f; background: rgba(42,157,143,.10); padding: .5rem .8rem;
       border-radius: 6px; margin-bottom: .45rem;}
.small {font-size: .85rem; opacity: .8;}
.steps {display: flex; gap: .5rem; margin: 0 0 1rem 0; flex-wrap: wrap;}
.step {flex: 1; min-width: 120px; padding: .45rem .7rem; border-radius: 999px; font-size: .9rem;
       border: 1px solid rgba(128,128,128,.35); opacity: .65; text-align: center;}
.step span {display: inline-block; width: 1.5rem; height: 1.5rem; line-height: 1.5rem; border-radius: 50%;
            margin-right: .45rem; background: rgba(128,128,128,.25); font-weight: 600;}
.step.now {opacity: 1; border-color: #2a9d8f; background: rgba(42,157,143,.12); font-weight: 600;}
.step.now span {background: #2a9d8f; color: white;}
.step.done {opacity: 1; border-color: #2a9d8f;}
.step.done span {background: #2a9d8f; color: white;}
.tip {border-left: 4px solid #e9a23b; background: rgba(233,162,59,.12); padding: .6rem .9rem; border-radius: 6px;}
</style>
"""

PIPELINE_DOT = """
digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#e8f4f2",
  color="#2a9d8f", fontname="Helvetica", fontsize=11]; edge [color="#264653"];
  cam [label="Camera /\\nVideo"]; mn [label="MoveNet\\n17 keypoints"]; nm [label="Normalise\\n+ smooth"];
  an [label="Joint\\nangles"]; cl [label="Pose\\nclassifier\\n(rules)"]; al [label="Alignment\\n+ corrections"];
  sc [label="Score +\\nstability"]; hd [label="Hold\\ntimer"]; db [label="Session\\nhistory"];
  pr [label="Personalise +\\nrecommend"];
  cam -> mn -> nm -> an -> cl -> al -> sc -> hd -> db -> pr; }
"""


# --------------------------------------------------------------------------- shared resources
@st.cache_resource
def get_db():
    return get_database()


@st.cache_resource(show_spinner="Preparing the MoveNet model (first start only)...")
def prepare_model(variant: str) -> str:
    return str(ensure_model(variant))


def make_detector() -> MoveNetDetector:
    variant = st.session_state.model_variant
    try:
        return MoveNetDetector(variant, model_file=prepare_model(variant))
    except Exception as exc:  # noqa: BLE001
        st.error("The MoveNet model could not be loaded.")
        st.code(str(exc))
        st.info(f"Download the model once and place it at `{model_path(variant)}` "
                "(see README -> *MoveNet model file*), or run the Colab notebook, which downloads it for you.")
        st.stop()


def init_state() -> None:
    defaults = dict(page="Home", user_name="guest", pose="T Pose", variant="both", just_saved=False,
                    manual_level="Beginner", auto_level=True, mirror=True,
                    model_variant=get_setting("MOVENET_VARIANT", "lightning"),
                    last_summary=None, last_frames=None, pending_profile=None, last_notes=[])
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def goto(page: str) -> None:
    st.session_state.page = page


def navigate(page: str) -> None:
    """Switch page from normal script code (not a button callback): applied at the top of the next run."""
    st.session_state["_goto"] = page
    st.rerun()


def select_pose(name: str) -> None:
    """Button callback: choose a pose and reset the side to a valid default."""
    spec = get_pose(name)
    st.session_state.pose = name
    st.session_state.variant = "auto" if spec.is_sided else spec.variant_names[0]


STEPS = [("Choose pose", "Pose Selection"), ("Calibrate (optional)", "Calibration"),
         ("Train", "Training"), ("Results", "Session Results")]

SIDE_HELP = {
    "Tree Pose": "which leg you are standing on",
    "Warrior II": "which leg is in front (the bent knee)",
    "Triangle Pose": "which hand reaches down towards your shin",
}


def stepper(current: str) -> None:
    idx = next((i for i, (_, page) in enumerate(STEPS) if page == current), -1)
    cells = []
    for i, (label, _) in enumerate(STEPS):
        cls = "done" if i < idx else ("now" if i == idx else "todo")
        mark = "\u2713" if i < idx else str(i + 1)
        cells.append(f'<div class="step {cls}"><span>{mark}</span>{label}</div>')
    st.markdown(f'<div class="steps">{"".join(cells)}</div>', unsafe_allow_html=True)


def nav_bar(items, key: str) -> None:
    """Row of navigation buttons. ``items`` = [(label, target_page, primary?), ...]."""
    st.write("")
    cols = st.columns(len(items))
    for i, (col, (label, target, primary)) in enumerate(zip(cols, items)):
        col.button(label, on_click=goto, args=(target,), key=f"nav_{key}_{i}",
                   type="primary" if primary else "secondary", use_container_width=True)


def auto_explainer(pose: str) -> None:
    spec = get_pose(pose)
    if not spec.is_sided:
        st.caption("This pose is symmetric - there is no side to choose.")
        return
    st.markdown(
        f"**What does 'Auto-detect' do?** For {pose} the trainer works out **{SIDE_HELP.get(pose, 'which side you use')}** "
        "by itself and judges you against that side's targets. Just get into the pose the way you like, hold it for a "
        "second or two and the detected side appears in the live panel. Choose a specific side only if you want to "
        "practise or calibrate that side on purpose.")


def pose_guide(pose: str, variant: str, title: str = "How it should look") -> None:
    """Reference pictures with target angles. Auto-detect on a sided pose shows both sides."""
    spec = get_pose(pose)
    variants = spec.variant_names if (variant == "auto" or variant not in spec.variants) else [variant]
    st.markdown(f"**{title}: {pose}**")
    cols = st.columns(len(variants))
    for col, v in zip(cols, variants):
        col.image(render_reference(pose, v, mirror=st.session_state.mirror), use_container_width=True,
                  caption=spec.variant_labels.get(v, v) if len(variants) > 1 else None)
    st.caption(reference_caption(st.session_state.mirror))
    with st.expander("Step-by-step instructions"):
        for i, step in enumerate(spec.instructions, 1):
            st.markdown(f"{i}. {step}")
        st.markdown(f"\U0001F4F7 *{spec.camera_hint}*")


def current_level(db, uid, pose) -> str:
    if st.session_state.auto_level:
        return db.get_levels(uid).get(pose, "Beginner")
    return st.session_state.manual_level


def live_params(db, uid, mode: str) -> dict:
    pose = st.session_state.pose
    return {"pose": pose, "variant": st.session_state.variant, "level": current_level(db, uid, pose),
            "calibrations": db.get_calibrations(uid, pose), "mirror": st.session_state.mirror,
            "model_variant": st.session_state.model_variant, "mode": mode}


def build_pipeline(db, uid, source: str, mode: str = "train") -> YogaPipeline:
    p = live_params(db, uid, mode)
    pipe = YogaPipeline(make_detector(), p["pose"], p["variant"], p["level"], p["calibrations"], source=source)
    if mode == "calibrate":
        pipe.start_calibration()
    else:
        pipe.start_session(source)
    return pipe


def show_fig(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- sidebar
def sidebar(db):
    st.sidebar.markdown("## \U0001F9D8 Yoga Trainer")
    st.sidebar.text_input("Your name", key="user_name", help="Sessions and calibration are stored per name.")
    uid = db.get_or_create_user(st.session_state.user_name)
    st.sidebar.radio("Navigate", PAGES, key="page")
    with st.sidebar.expander("Settings"):
        st.selectbox("MoveNet model", ["lightning", "thunder"], key="model_variant",
                     help="Lightning is faster (recommended); Thunder is more accurate but slower.")
        st.checkbox("Mirror webcam view", key="mirror")
        st.checkbox("Adaptive difficulty (automatic)", key="auto_level",
                    help="Levels go up or down per pose from your recent sessions.")
        if not st.session_state.auto_level:
            st.selectbox("Difficulty", LEVEL_ORDER, key="manual_level")
    with st.sidebar.expander("Backup & data"):
        st.caption(f"Storage: {db.describe()}")
        st.download_button("Download my data (JSON)", json.dumps(db.export_user_json(uid)),
                           file_name=f"yoga_backup_{st.session_state.user_name}.json", mime="application/json")
        up = st.file_uploader("Restore from backup", type=["json"], key="restore_file")
        if up is not None and st.button("Restore now"):
            try:
                added = db.import_user_json(uid, json.load(up))
                st.success(f"Restored {added['sessions_added']} new sessions.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not restore: {exc}")
    st.sidebar.caption("Not medical advice. Stop if you feel pain; practise on a clear, safe floor.")
    return uid


# --------------------------------------------------------------------------- pages
def page_home(db, uid):
    st.markdown(f'<div class="hero"><h1>\U0001F9D8 {APP_TITLE}</h1><p>{APP_SUBTITLE}</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><h4>1. Calibrate</h4>Hold a pose as well as you comfortably can. '
                'Target ranges adapt (within safe limits) to your flexibility and proportions.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><h4>2. Train</h4>Use your webcam or upload a video. Get a live skeleton, '
                'scores and joint-level corrections. The timer runs only while your pose is correct and steady.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><h4>3. Improve</h4>Review results, track progress, spot errors that repeat '
                'across sessions and follow personalised recommendations.</div>', unsafe_allow_html=True)
    st.write("")
    b1, b2, b3 = st.columns(3)
    b1.button("\u25B6 Start: choose a pose", on_click=goto, args=("Pose Selection",), type="primary", use_container_width=True)
    b2.button("Calibrate", on_click=goto, args=("Calibration",), use_container_width=True)
    b3.button("Go straight to training", on_click=goto, args=("Training",), use_container_width=True)
    with st.expander("How to use this app (30 seconds)", expanded=True):
        st.markdown(
            "1. **Choose a pose** - you will see how it should look.\n"
            "2. **Calibrate** (optional, once per pose) - hold a comfortable version so the targets fit your body.\n"
            "3. **Train** - press *Start*, allow the camera, step back so your whole body is visible and copy the picture "
            "next to the video. Follow the red-highlighted joints and the correction messages. The timer only runs while "
            "you are correct *and* steady.\n"
            "4. Press **Finish & save** - you are taken to your **results**, and your history, recurring errors and "
            "recommendations build up over time.")

    st.subheader("How it works")
    st.graphviz_chart(PIPELINE_DOT)
    sessions = db.get_sessions(uid)
    if len(sessions):
        s = overall_summary(sessions)
        st.subheader(f"Welcome back, {st.session_state.user_name}")
        m = st.columns(4)
        m[0].metric("Sessions", s["sessions"])
        m[1].metric("Best score", f"{s['best_score']:.0f}")
        m[2].metric("Time held correctly", f"{s['total_hold_min']:.1f} min")
        m[3].metric("Practice streak", f"{s['streak_days']} days")
    with st.expander("Method, transparency and limitations"):
        st.markdown(
            "- **Pose estimation:** Google's MoveNet (TensorFlow Lite) returns 17 body keypoints per frame.\n"
            "- **Pose classification is rule-based, not a trained classifier.** Every target range is a visible, "
            "editable number in `ai/yoga/poses.py`. No accuracy percentage is claimed.\n"
            "- **Scores** combine alignment (how many joints are inside their target range), stability (how still "
            "you are) and, for symmetric poses, left/right symmetry.\n"
            "- **Corrections** always show the current angle, the target range, the deviation and what to change.\n"
            "- **Limits:** angles are measured in 2-D from one camera, so camera placement matters (front view for most "
            "poses, side view for Chair Pose). Loose clothing, poor light or a cropped body reduce accuracy.\n"
            "- This is an educational project, not medical or professional yoga instruction.")


def criteria_dataframe(pose: str, variant: str, level: str, calibration) -> pd.DataFrame:
    base = criteria_for(pose, variant)
    final = apply_difficulty(apply_calibration(base, calibration), level)
    rows = []
    for b, f in zip(base, final):
        rows.append({"Joint / rule": f.label, "Default target": format_range(b.low, b.high, b.unit),
                     "Your target": format_range(f.low, f.high, f.unit), "Weight": f.weight,
                     "Defines the pose": "yes" if f.signature else ""})
    return pd.DataFrame(rows)


def page_pose_selection(db, uid):
    stepper("Pose Selection")
    st.header("Step 1 - Choose your pose")
    st.write("Click a pose to select it, check how it should look, then press **Continue**.")
    selected = st.session_state.pose

    cols = st.columns(len(list_poses()))
    for col, name in zip(cols, list_poses()):
        sp = get_pose(name)
        col.image(render_reference(name, sp.variant_names[0], width=240, height=310,
                                   mirror=st.session_state.mirror, show_targets=False), use_container_width=True)
        col.button(("\u2713 " if name == selected else "") + name, key=f"pick_{name}", on_click=select_pose,
                   args=(name,), type="primary" if name == selected else "secondary", use_container_width=True)

    pose = st.session_state.pose
    spec = get_pose(pose)
    opts = ["auto"] + spec.variant_names if spec.is_sided else spec.variant_names
    if st.session_state.variant not in opts:
        st.session_state.variant = opts[0]
    labels = {"auto": "Auto-detect (recommended)", **spec.variant_labels}

    st.divider()
    left, right = st.columns([2, 3])
    with left:
        st.subheader(f"{spec.name}  \u00b7  *{spec.sanskrit}*")
        st.write(spec.description)
        if spec.is_sided:
            # not keyed on purpose: Streamlit deletes the state of keyed widgets on pages where they are not drawn
            st.session_state.variant = st.radio("Which side?", opts, index=opts.index(st.session_state.variant),
                                                format_func=lambda v: labels.get(v, v))
        auto_explainer(pose)
        level = current_level(db, uid, pose)
        st.markdown(f"**Difficulty:** {level}  \n<span class='small'>{LEVELS[level].description}</span>", unsafe_allow_html=True)
        st.markdown(f'<div class="tip">\U0001F4F7 {spec.camera_hint}</div>', unsafe_allow_html=True)
        if spec.benefits:
            st.caption(f"Benefits: {spec.benefits}")
    with right:
        pose_guide(pose, st.session_state.variant, "How it should look")
        show_variant = st.session_state.variant if st.session_state.variant in spec.variants else spec.variant_names[0]
        with st.expander("Exact target ranges for this pose (transparent rules)", expanded=False):
            cal = db.get_calibrations(uid, pose).get(show_variant)
            st.dataframe(criteria_dataframe(pose, show_variant, level, cal), hide_index=True)
            st.caption("'Your target' includes the difficulty level and, if saved, your calibration.")

    nav_bar([("\u2190 Home", "Home", False),
             ("Continue: calibrate to my body \u2192", "Calibration", True),
             ("Skip calibration: start training \u2192", "Training", False)], "pose")


def profile_table(pose, variant, profile) -> pd.DataFrame:
    rows = []
    for c in criteria_for(pose, variant):
        stat = profile["metrics"].get(c.metric)
        if not stat:
            continue
        val = stat["median"]
        inside = c.low <= val <= c.high
        rows.append({"Joint / rule": c.label, "Your comfortable value": format_value(val, c.unit),
                     "Default target": format_range(c.low, c.high, c.unit),
                     "Result": "inside default range" if inside else "outside - range will be extended (limited)"})
    return pd.DataFrame(rows)


def calibration_variant_choice(pose: str):
    spec = get_pose(pose)
    if spec.is_sided:
        return st.radio("Side to calibrate", spec.variant_names, format_func=lambda v: spec.variant_labels[v], horizontal=True)
    return spec.variant_names[0]


def page_calibration(db, uid):
    pose = st.session_state.pose
    stepper("Calibration")
    st.header("Step 2 - Calibrate to your body (optional)")
    spec = get_pose(pose)
    st.write(f"Pose: **{pose}**. Hold your most comfortable, best-effort version of the pose for 5-10 seconds. "
             "The target ranges are then widened (never tightened, and by a limited amount) to fit your flexibility, "
             "and stance targets are scaled to your leg length. **Once per pose is enough - you can skip this step.**")
    top = st.columns([1, 1, 2])
    top[0].button("\u2190 Back to pose selection", on_click=goto, args=("Pose Selection",), key="cal_back")
    top[1].button("Skip: start training \u2192", on_click=goto, args=("Training",), key="cal_skip", type="primary")
    st.warning("Calibrate with a genuine attempt at the pose. It personalises the *limits* of the ranges - it does not "
               "certify that a position is safe for your body.")
    variant = calibration_variant_choice(pose)
    guide_col, work_col = st.columns([2, 3])
    with guide_col:
        pose_guide(pose, variant, "Copy this pose")
    with work_col:
        calibration_workspace(db, uid, pose, variant, spec)
    nav_bar([("\u2190 Back to pose selection", "Pose Selection", False),
             ("Continue: start training \u2192", "Training", True)], "cal")


def calibration_workspace(db, uid, pose, variant, spec):
    saved = db.get_calibrations(uid, pose)
    if saved:
        with st.expander(f"Saved calibration for {pose}", expanded=False):
            for v, prof in saved.items():
                st.markdown(f"**{spec.variant_labels.get(v, v)}** - {prof['n_samples']} samples, {prof['created_at']}")
                rows = describe_adaptation(criteria_for(pose, v), apply_calibration(criteria_for(pose, v), prof))
                st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame({"Info": ["No ranges needed changing."]}), hide_index=True)
            if st.button("Delete calibration for this pose"):
                db.delete_calibrations(uid, pose)
                st.rerun()

    tabs = st.tabs(["Webcam", "Upload video", "Photo"])
    profile = None
    params = {"pose": pose, "variant": variant, "level": "Intermediate", "calibrations": None,
              "mirror": st.session_state.mirror, "model_variant": st.session_state.model_variant, "mode": "calibrate"}

    with tabs[0]:
        if not WEBRTC_AVAILABLE:
            st.warning("`streamlit-webrtc` is not installed - use the video or photo tab.")
        else:
            ctx = webrtc_streamer(key=f"calib-{pose}-{variant}", mode=WebRtcMode.SENDRECV,
                                  rtc_configuration={"iceServers": get_ice_servers()},
                                  media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                                  video_processor_factory=YogaProcessor, async_processing=True)
            if ctx.video_processor:
                ctx.video_processor.set_params(params)
            c1, c2 = st.columns(2)
            finish = c1.button("Finish calibration", type="primary", key="calib_finish")
            if c2.button("Restart recording", key="calib_restart") and ctx.video_processor:
                ctx.video_processor.restart()
            info = st.empty()
            if finish and ctx.video_processor and ctx.video_processor.pipeline and ctx.video_processor.pipeline.calibrator:
                try:
                    profile = ctx.video_processor.pipeline.calibrator.build_profile()
                except ValueError as exc:
                    st.error(str(exc))
            elif ctx.state.playing and ctx.video_processor:
                while ctx.state.playing:
                    pipe = ctx.video_processor.pipeline
                    n = pipe.calibrator.n_samples if pipe and pipe.calibrator else 0
                    info.progress(min(1.0, n / 60), text=f"Recording... {n} frames (aim for 60+; hold still in your pose)")
                    time.sleep(0.4)

    with tabs[1]:
        vid = st.file_uploader("Video of you holding the pose", type=["mp4", "mov", "avi", "mkv", "webm"], key="calib_video")
        if vid is not None and st.button("Calibrate from video"):
            pipe = YogaPipeline(make_detector(), pose, variant, "Intermediate", None, source="video")
            pipe.start_calibration()
            analyze_video(vid, pipe, max_seconds=30)
            try:
                profile = pipe.calibrator.build_profile()
            except ValueError as exc:
                st.error(str(exc))

    with tabs[2]:
        shot = st.camera_input("Take a photo of your comfortable pose", key="calib_photo")
        if shot is not None and st.button("Calibrate from photo"):
            img = decode_image(shot.getvalue())
            pipe = YogaPipeline(make_detector(), pose, variant, "Intermediate", None, source="photo")
            pipe.start_calibration()
            res = pipe.process_frame(img, 0.0)
            if res.detected:
                st.image(bgr_to_rgb(annotate_frame(img, res)))
                profile = pipe.calibrator.build_profile()
                st.caption("A single photo gives only one sample - a short video or webcam recording is more reliable.")
            else:
                st.error(res.message)

    if profile is not None:
        st.session_state.pending_profile = (pose, variant, profile)
    pending = st.session_state.pending_profile
    if pending and pending[0] == pose and pending[1] == variant:
        _, _, prof = pending
        st.subheader("Calibration result")
        st.write(f"{prof['n_samples']} frames recorded.")
        st.dataframe(profile_table(pose, variant, prof), hide_index=True)
        adapted = describe_adaptation(criteria_for(pose, variant), apply_calibration(criteria_for(pose, variant), prof))
        if adapted:
            st.markdown("**Ranges that will change:**")
            st.dataframe(pd.DataFrame(adapted), hide_index=True)
        else:
            st.success("Your values are inside the default ranges - nothing needs to change.")
        if st.button("Save calibration", type="primary"):
            db.save_calibration(uid, pose, variant, prof)
            st.session_state.pending_profile = None
            st.success("Calibration saved. It is applied automatically in Training - press **Continue** below.")


# --------------------------------------------------------------------------- analysis helpers
def analyze_video(file, pipeline: YogaPipeline, max_seconds: float = 90.0):
    """Run a pipeline over an uploaded video; returns (best_frame, worst_frame) annotated RGB images."""
    suffix = os.path.splitext(file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getbuffer())
        path = tmp.name
    best = worst = None
    best_s, worst_s = -1.0, 101.0
    try:
        info = probe_video(path)
        total = min(info.duration_s or max_seconds, max_seconds)
        bar = st.progress(0.0, text="Analysing video...")
        slot = st.empty()
        for i, (frame, t) in enumerate(iter_video_frames(path, 10.0, max_seconds)):
            res = pipeline.process_frame(frame, t)
            if res.detected and np.isfinite(res.overall):
                if res.overall > best_s or res.overall < worst_s or i % 5 == 0:
                    shown = bgr_to_rgb(annotate_frame(frame, res))
                    if res.overall > best_s:
                        best_s, best = res.overall, shown
                    if res.overall < worst_s:
                        worst_s, worst = res.overall, shown
            if i % 4 == 0:
                slot.image(bgr_to_rgb(annotate_frame(frame, res)), caption=f"t = {t:.1f}s")
            bar.progress(min(1.0, t / max(total, 1e-6)), text=f"Analysing video... {t:.0f}s / {total:.0f}s")
        bar.empty()
    finally:
        os.unlink(path)
    return best, worst


def render_live(container, res) -> None:
    """Right-hand live panel: scores, hold timer and joint corrections."""
    with container.container():
        if res is None:
            st.info("Waiting for the camera...")
            return
        c = st.columns(4)
        c[0].metric("Score", "-" if not res.detected else f"{res.overall:.0f}")
        c[1].metric("Alignment", "-" if not res.detected else f"{res.alignment.score:.0f}")
        c[2].metric("Stability", "-" if not res.detected else f"{res.stability:.0f}")
        hold = res.hold or {}
        c[3].metric("Held", f"{hold.get('held_s', 0):.1f}s")
        st.progress(float(hold.get("progress", 0.0)),
                    text=("⏱ Timer running" if hold.get("active") else "⏸ Timer paused") +
                         f" - target {hold.get('target_s', 0):.0f}s")
        st.caption(f"Detected: **{res.detected_pose}**  ·  Target: **{res.target_pose}**"
                   + (f" ({res.variant})" if res.variant and res.variant != "both" else ""))
        if res.corrections:
            for corr in res.corrections:
                imp = impact_for(res.target_pose, corr.key, corr.direction, corr.label)
                why = f'<br><small>⚠ <b>Why it matters</b> - {imp.live_text}</small>' if imp else ""
                st.markdown(f'<div class="corr">{corr.text}{why}</div>', unsafe_allow_html=True)
        elif res.detected and res.is_correct:
            st.markdown('<div class="good">Great posture - keep holding steady.</div>', unsafe_allow_html=True)
        else:
            st.write(res.message)


def finalize_session(db, uid, summary: dict, frames=None) -> None:
    pose = summary["pose"]
    if summary["detected_frames"] < 10:
        st.error("Too few frames with a detected body - nothing was saved. Check lighting and make sure your whole body is visible.")
        return
    db.save_session(uid, summary)
    st.session_state.last_summary = summary
    st.session_state.last_frames = frames
    notes = session_feedback(summary)
    if st.session_state.auto_level:
        history = db.get_sessions(uid, pose)
        cur = db.get_levels(uid).get(pose, "Beginner")
        new, why = recommend_level(history, cur)
        if new != cur:
            db.set_level(uid, pose, new)
            notes.append(f"Difficulty for {pose} changed {cur} -> {new}. {why}")
        else:
            db.set_level(uid, pose, cur)
    st.session_state.last_notes = notes
    st.session_state.just_saved = True
    navigate("Session Results")


def page_training(db, uid):
    pose, variant = st.session_state.pose, st.session_state.variant
    spec = get_pose(pose)
    level = current_level(db, uid, pose)
    cfg = LEVELS[level]
    stepper("Training")
    st.header("Step 3 - Train")
    m = st.columns(4)
    m[0].metric("Pose", pose)
    m[1].metric("Side", {"auto": "Auto-detect"}.get(variant, spec.variant_labels.get(variant, variant)))
    m[2].metric("Difficulty", level)
    m[3].metric("Hold target", f"{cfg.hold_target_s:.0f}s" + (" (one go)" if cfg.strict_hold else ""))
    with st.expander("How to train - 5 steps", expanded=True):
        st.markdown(
            "1. Look at the **picture next to the camera** - copy that shape.\n"
            "2. Press **Start** on the video, allow camera access and step back until your **whole body** is visible "
            f"({spec.camera_hint.split(' - ')[0].lower()}).\n"
            "3. Get into the pose. Joints that are off turn **red** on the skeleton and a correction message appears "
            "(current angle, target range, what to change).\n"
            "4. The **hold timer** runs only while you are correct *and* steady - reach the target time.\n"
            "5. Press **Finish & save session** - you will be taken to your results.")
        if variant == "auto" and spec.is_sided:
            auto_explainer(pose)
    if db.get_calibrations(uid, pose):
        st.caption("\u2705 Personal calibration is active for this pose.")
    top = st.columns([1, 1, 2])
    top[0].button("\u2190 Change pose", on_click=goto, args=("Pose Selection",), key="train_back")

    tabs = st.tabs(["Live webcam", "Upload video", "Photo snapshot", "Demo (no camera)"])

    with tabs[0]:
        if not WEBRTC_AVAILABLE:
            st.warning("`streamlit-webrtc` is not installed - use the video upload tab.")
        else:
            video_col, panel_col = st.columns([3, 2])
            with panel_col:
                pose_guide(pose, variant, "Copy this pose")
                panel = st.empty()
            with video_col:
                ctx = webrtc_streamer(key=f"train-{pose}", mode=WebRtcMode.SENDRECV,
                                      rtc_configuration={"iceServers": get_ice_servers()},
                                      media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                                      video_processor_factory=YogaProcessor, async_processing=True)
                b1, b2 = st.columns(2)
                finish = b1.button("Finish & save session", type="primary", key="train_finish")
                restart = b2.button("Restart session", key="train_restart")
            if ctx.video_processor:
                ctx.video_processor.set_params(live_params(db, uid, "train"))
                if restart:
                    ctx.video_processor.restart()
            if finish:
                proc = ctx.video_processor
                if proc and proc.pipeline and proc.pipeline.recorder:
                    summary = proc.pipeline.recorder.summary()
                    proc.restart()          # before finalize_session, which ends the run by navigating away
                    finalize_session(db, uid, summary)
                else:
                    st.warning("Start the camera first, practise the pose, then press Finish.")
            elif ctx.state.playing and ctx.video_processor:
                st.caption("Press **Start** above and allow camera access. If the video never appears, your network may block "
                           "WebRTC - use the video upload or photo tab.")
                while ctx.state.playing:
                    if ctx.video_processor.error:
                        panel.error(ctx.video_processor.error)
                    else:
                        render_live(panel, ctx.video_processor.get_latest())
                    time.sleep(0.3)
            else:
                st.caption("Press **Start**, allow camera access and step back until your whole body is visible.")

    with tabs[1]:
        up_col, up_guide = st.columns([3, 2])
        with up_guide:
            pose_guide(pose, variant, "Copy this pose")
        with up_col:
            st.write("Film yourself from the front (Chair Pose: from the side), whole body in view, 10-60 seconds.")
            vid = st.file_uploader("Upload a video of your practice (MP4 recommended)", type=["mp4", "mov", "avi", "mkv", "webm"], key="train_video")
            max_s = st.slider("Analyse at most (seconds)", 10, 180, 90, key="train_max_s")
        if vid is not None and st.button("Analyse video", type="primary", key="train_video_go"):
            pipe = build_pipeline(db, uid, "video")
            best, worst = analyze_video(vid, pipe, float(max_s))
            finalize_session(db, uid, pipe.recorder.summary(), {"best": best, "worst": worst})

    with tabs[2]:
        ph_col, ph_guide = st.columns([3, 2])
        with ph_guide:
            pose_guide(pose, variant, "Copy this pose")
        with ph_col:
            shot = st.camera_input("Take a photo while holding the pose", key="train_photo")
        if shot is not None:
            img = decode_image(shot.getvalue())
            pipe = build_pipeline(db, uid, "photo")
            res = pipe.process_frame(img, 0.0)
            c1, c2 = st.columns([3, 2])
            c1.image(bgr_to_rgb(annotate_frame(img, res)))
            render_live(c2.empty(), res)
            st.caption("A photo gives instant feedback but has no motion, so the hold timer and stability need video.")

    with tabs[3]:
        st.write("No camera or model? Run the full pipeline on a **synthetic stick figure** of the selected pose. "
                 "This is only a demonstration of the scoring / dashboards - it is not a real person.")
        save_demo = st.checkbox("Save the demo to my history (marked 'synthetic-demo')", value=False)
        if st.button("Run demo session", key="demo_go"):
            demo_variant = variant if variant in spec.variants else spec.variant_names[0]
            coords = make_pose(pose, demo_variant)
            pipe = YogaPipeline(None, pose, variant, level, db.get_calibrations(uid, pose), source="synthetic-demo")
            pipe.start_session("synthetic-demo")
            rng = np.random.default_rng(7)
            for i in range(int(25 * 15)):
                pipe.analyze_keypoints(to_pixels(coords, noise_px=1.0, rng=rng), i / 15.0)
            summary = pipe.recorder.summary()
            if save_demo:
                finalize_session(db, uid, summary)
            else:
                st.session_state.last_summary = summary
                st.session_state.last_frames = None
                st.session_state.just_saved = False
                navigate("Session Results")


def page_results(db, uid):
    stepper("Session Results")
    st.header("Step 4 - Session results")
    if st.session_state.just_saved:
        st.success("Session saved \u2713 - here is how you did.")
        st.session_state.just_saved = False
    sessions = db.get_sessions(uid)
    options = ["Latest session (this visit)"] if st.session_state.last_summary else []
    ids = sessions["id"].tolist()[::-1]
    labels = {i: f"#{i} - {sessions.loc[sessions['id'] == i, 'pose'].iloc[0]} - "
                 f"{sessions.loc[sessions['id'] == i, 'started_at'].iloc[0][:16].replace('T', ' ')}" for i in ids}
    if not options and not ids:
        st.info("No sessions yet. Complete a training session first.")
        nav_bar([("Go to training \u2192", "Training", True)], "res_empty")
        return
    choice = st.selectbox("Session", options + [labels[i] for i in ids])
    if choice.startswith("Latest"):
        s = st.session_state.last_summary
        frames = st.session_state.last_frames
    else:
        sid = [i for i in ids if labels[i] == choice][0]
        s, frames = db.get_session(int(sid)), None

    st.subheader(f"{s['pose']}  ·  {s.get('difficulty', '')}")
    m = st.columns(5)
    m[0].metric("Overall score", f"{s['avg_score']:.0f}", grade_for(s["avg_score"]))
    m[1].metric("Alignment", f"{s['avg_alignment']:.0f}")
    m[2].metric("Stability", f"{s['avg_stability']:.0f}")
    m[3].metric("Symmetry", f"{s['avg_symmetry']:.0f}")
    m[4].metric("Held correctly", f"{s['held_s']:.1f}s", "target reached" if s["completed"] else f"target {s['hold_target_s']:.0f}s")
    st.caption(f"Duration {s['duration_s']:.0f}s  ·  longest uninterrupted hold {s['best_streak_s']:.1f}s  ·  "
               f"{s['breaks']} interruptions  ·  source: {s.get('source', '')}")
    if choice.startswith("Latest") and st.session_state.last_notes:
        for n in st.session_state.last_notes:
            st.write("• " + n)
    c1, c2 = st.columns(2)
    with c1:
        show_fig(timeline_figure(s["timeline"], s["hold_target_s"]))
    with c2:
        show_fig(joint_error_figure(s["errors"]))
    if s["errors"]:
        st.subheader("Joint-level analysis")
        df = pd.DataFrame(s["errors"])
        view = pd.DataFrame({
            "Joint": df["label"],
            "Out of range": (df["bad_frac"] * 100).round(0).astype(int).astype(str) + "% of frames",
            "Typical deviation": df.apply(lambda r: f"{r['mean_deviation']:+.0f}°" if r["unit"] == "°" else f"{r['mean_deviation']:+.2f}", axis=1),
            "Usually": df.apply(lambda r: describe_direction(r["direction"], r["unit"]) if r["bad_frac"] > 0.05 else "in range", axis=1),
        })
        st.dataframe(view, hide_index=True)
    impacts = session_impacts(s["pose"], s["errors"])
    if impacts:
        st.subheader("Impact of the mistakes in this session")
        st.caption("What the joints that were out of range for a large part of this session can do to the pose and to your body.")
        for it in impacts:
            tag = "Strain risk" if it["kind"] == "strain" else "Reduces the benefit"
            st.markdown(
                f'<div class="corr"><b>{it["joint"]}</b> - out of range for {it["bad_pct"]}% of the session '
                f'<i>({tag}: {it["area"]})</i><br>{it["effect"]}<br><b>If repeated:</b> {it["long_term"]}</div>',
                unsafe_allow_html=True)
        st.caption(IMPACT_DISCLAIMER)
    if frames and (frames.get("best") is not None):
        f1, f2 = st.columns(2)
        f1.image(frames["best"], caption="Best moment")
        f2.image(frames["worst"], caption="Needs most work")
    st.subheader("What next?")
    nav_bar([("Practise this pose again", "Training", True), ("Try another pose", "Pose Selection", False),
             ("See my progress", "Progress Dashboard", False), ("Recommendations", "Recommendations", False)], "res")


def page_progress(db, uid):
    st.header("Progress dashboard")
    df = db.get_sessions(uid)
    if df.empty:
        st.info("No sessions yet - your progress will appear here after your first training session.")
        return
    poses = ["All poses"] + sorted(df["pose"].unique())
    pick = st.selectbox("Pose", poses)
    if pick != "All poses":
        df = df[df["pose"] == pick]
    p = prepare_sessions(df)
    s = overall_summary(df)
    m = st.columns(5)
    m[0].metric("Sessions", s["sessions"])
    m[1].metric("Best score", f"{s['best_score']:.0f}")
    m[2].metric("Average score", f"{s['avg_score']:.0f}", f"{s['improvement']:+.1f} vs previous 5" if s["sessions"] > 5 else None)
    m[3].metric("Hold target reached", f"{s['completion_rate']:.0f}%")
    m[4].metric("Streak", f"{s['streak_days']} days")
    c1, c2 = st.columns(2)
    with c1:
        show_fig(progress_figure(p))
    with c2:
        show_fig(hold_time_figure(p))
    st.subheader("Stability and symmetry trend")
    st.line_chart(p.set_index("session_no")[["avg_alignment", "avg_stability", "avg_symmetry"]])
    st.subheader("Per-pose summary")
    st.dataframe(per_pose_summary(df), hide_index=True)
    st.subheader("Weekly summary")
    st.dataframe(weekly_summary(df), hide_index=True)
    with st.expander("All sessions"):
        st.dataframe(p.drop(columns=["date"], errors="ignore"), hide_index=True)
        st.download_button("Download sessions (CSV)", p.drop(columns=["date"], errors="ignore").to_csv(index=False),
                           file_name="yoga_sessions.csv", mime="text/csv")


def page_recurring(db, uid):
    st.header("Recurring errors")
    st.caption("A joint is flagged when it was outside its target range for 30% or more of a session, in at least two of "
               "your last six sessions of that pose (and at least half of the sessions where it was measured).")
    rec = detect_recurring_errors(db.get_errors(uid))
    if rec.empty:
        st.success("No recurring errors detected yet. Keep practising - this analysis needs at least two sessions per pose.")
        return
    show_fig(recurring_figure(rec))
    for _, r in rec.iterrows():
        st.markdown(
            f'<div class="corr"><b>{r["pose"]} - {r["label"]}</b><br>Flagged in {r["sessions_affected"]} of the last '
            f'{r["sessions_considered"]} sessions (about {r["avg_bad_pct"]:.0f}% of frames); {describe_direction(r["direction"], r["unit"])}. '
            f'<br><i>Tip: {r["tip"]}</i>'
            + (lambda imp: (f'<br><b>Why it matters:</b> {imp.effect} <b>If this continues:</b> {imp.long_term}' if imp else "")
               )(impact_for(r["pose"], r["key"], r["direction"], r["label"]))
            + '</div>', unsafe_allow_html=True)
    st.caption(IMPACT_DISCLAIMER)


def page_recommendations(db, uid):
    st.header("Recommendations")
    sessions = db.get_sessions(uid)
    rec = detect_recurring_errors(db.get_errors(uid))
    levels = db.get_levels(uid)
    recs = build_recommendations(sessions, rec, levels)
    for r in recs:
        with st.container(border=True):
            st.markdown(f"**{r.category}: {r.title}**")
            st.write(r.detail)
            st.caption(f"Why: {r.reason}")
    st.subheader("Difficulty by pose")
    cols = st.columns(len(list_poses()))
    for col, pose in zip(cols, list_poses()):
        with col:
            cur = levels.get(pose, "Beginner")
            new = st.selectbox(pose, LEVEL_ORDER, index=LEVEL_ORDER.index(cur), key=f"lvl_{pose}")
            if new != cur:
                db.set_level(uid, pose, new)
                st.rerun()
    st.caption("With adaptive difficulty on, levels also change automatically after sessions (promotion needs two "
               "sessions scoring 80+ with the hold completed).")


# --------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    if st.session_state.get("_goto"):
        st.session_state.page = st.session_state.pop("_goto")
    st.markdown(CSS, unsafe_allow_html=True)
    db = get_db()
    uid = sidebar(db)
    page = st.session_state.page
    {"Home": page_home, "Pose Selection": page_pose_selection, "Calibration": page_calibration,
     "Training": page_training, "Session Results": page_results, "Progress Dashboard": page_progress,
     "Recurring Errors": page_recurring, "Recommendations": page_recommendations}[page](db, uid)


main()
