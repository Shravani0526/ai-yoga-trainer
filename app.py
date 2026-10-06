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
.stButton > button {width: 100%; border-radius: 10px; font-weight: 600; padding: .55rem 1rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .5rem .8rem;}
.card {transition: box-shadow .15s;} .card:hover {box-shadow: 0 4px 14px rgba(0,0,0,.12);}
.steps {display: flex; gap: .4rem; margin: 0 0 1.1rem 0; flex-wrap: wrap;}
.step {flex: 1 1 120px; padding: .55rem .8rem; border-radius: 10px; font-size: .92rem; font-weight: 600;
       border: 1px solid rgba(128,128,128,.3); opacity: .6;}
.step.done {background: rgba(42,157,143,.12); border-color: #2a9d8f; opacity: 1;}
.step.active {background: #2a9d8f; color: white; border-color: #2a9d8f; opacity: 1;}
.tip {border-left: 4px solid #2a9d8f; background: rgba(42,157,143,.08); padding: .6rem .9rem;
      border-radius: 6px; margin: .4rem 0 .8rem 0;}
#MainMenu, footer {visibility: hidden;}
@media (max-width: 640px) {.hero h1 {font-size: 1.5rem;} .block-container {padding-left: .8rem; padding-right: .8rem;}}
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
    defaults = dict(page="Home", user_name="guest", pose="T Pose", variant="auto",
                    manual_level="Beginner", auto_level=True, mirror=True,
                    model_variant=get_setting("MOVENET_VARIANT", "lightning"),
                    last_summary=None, last_frames=None, pending_profile=None, last_notes=[],
                    auto_start=True, auto_finish=True, calib_notice=None, _goto=None)
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def goto(page: str) -> None:
    """Button callback: switch page (safe inside on_click)."""
    st.session_state.page = page


def goto_now(page: str) -> None:
    """Switch page from inside the running script (auto-advance). The sidebar radio owns `page`, so the
    change is queued and applied at the top of the next run (see main())."""
    st.session_state._goto = page
    st.rerun()


FLOW = [("Pose Selection", "1 · Choose pose"), ("Calibration", "2 · Calibrate"),
        ("Training", "3 · Train"), ("Session Results", "4 · Results")]


def stepper(current: str) -> None:
    """Progress bar for the guided flow: Pose -> Calibrate -> Train -> Results."""
    names = [n for n, _ in FLOW]
    cur = names.index(current)
    html = "".join(f'<div class="step {"active" if i == cur else "done" if i < cur else ""}">'
                   f'{"✓ " if i < cur else ""}{label}</div>' for i, (_, label) in enumerate(FLOW))
    st.markdown(f'<div class="steps">{html}</div>', unsafe_allow_html=True)


def webrtc_extra() -> dict:
    """Start the camera automatically (the browser still asks for permission once)."""
    return {"desired_playing_state": True} if st.session_state.auto_start else {}


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
    st.sidebar.caption(f"Current pose: **{st.session_state.pose}**")
    with st.sidebar.expander("Settings"):
        st.selectbox("MoveNet model", ["lightning", "thunder"], key="model_variant",
                     help="Lightning is faster (recommended); Thunder is more accurate but slower.")
        st.checkbox("Mirror webcam view", key="mirror")
        st.checkbox("Start camera automatically", key="auto_start")
        st.checkbox("Finish session automatically when hold target is reached", key="auto_finish")
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
    if st.session_state.user_name.strip().lower() in ("", "guest"):
        st.markdown('<div class="tip">Tip: enter your name in the sidebar so your progress is saved to you.</div>',
                    unsafe_allow_html=True)
    b1, b2 = st.columns([2, 1])
    b1.button("Start guided session  →", on_click=goto, args=("Pose Selection",), type="primary", key="home_start")
    b2.button("Quick start (skip setup)", on_click=goto, args=("Training",), key="home_quick")

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


def choose_pose(next_page: str) -> None:
    """Button callback: remember the chosen pose/side and move on to the next step automatically."""
    pose = st.session_state.get("sel_pose", st.session_state.pose)
    spec = get_pose(pose)
    default_variant = "auto" if spec.is_sided else spec.variant_names[0]
    st.session_state.pose = pose
    st.session_state.variant = st.session_state.get(f"sel_variant_{pose}", default_variant)
    st.session_state.page = next_page


def page_pose_selection(db, uid):
    stepper("Pose Selection")
    st.header("Choose your pose")
    left, right = st.columns([1, 2], gap="large")
    with left:
        poses = list_poses()
        pose = st.radio("Pose", poses, index=poses.index(st.session_state.pose), key="sel_pose")
        spec = get_pose(pose)
        opts = ["auto"] + spec.variant_names if spec.is_sided else spec.variant_names
        labels = {"auto": "Auto-detect side", **spec.variant_labels}
        cur = st.session_state.variant
        variant = st.radio("Side / variant", opts, format_func=lambda v: labels.get(v, v),
                           index=opts.index(cur) if cur in opts else 0, key=f"sel_variant_{pose}")
        level = current_level(db, uid, pose)
        st.markdown(f"**Difficulty:** {level}  \n<span class='small'>{LEVELS[level].description}</span>", unsafe_allow_html=True)
        st.button("Continue to calibration  →", type="primary", on_click=choose_pose, args=("Calibration",), key="pose_next")
        st.button("Skip calibration - start training", on_click=choose_pose, args=("Training",), key="pose_skip")
    with right:
        st.subheader(f"{spec.name}  ·  *{spec.sanskrit}*")
        st.write(spec.description)
        st.markdown("**How to do it**")
        for step in spec.instructions:
            st.markdown(f"- {step}")
        st.info(f"\U0001F4F7 {spec.camera_hint}")
        if spec.benefits:
            st.caption(f"Benefits: {spec.benefits}")
        show_variant = variant if variant in spec.variants else spec.variant_names[0]
        with st.expander("Target ranges used for this pose (transparent rules)", expanded=False):
            cal = db.get_calibrations(uid, pose).get(show_variant)
            st.dataframe(criteria_dataframe(pose, show_variant, level, cal), hide_index=True)
            st.caption("'Your target' includes the difficulty level and, if saved, your calibration.")


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
        cur = st.session_state.variant
        idx = spec.variant_names.index(cur) if cur in spec.variant_names else 0
        return st.radio("Side to calibrate", spec.variant_names, index=idx,
                        format_func=lambda v: spec.variant_labels[v], horizontal=True, key=f"calib_side_{pose}")
    return spec.variant_names[0]


CALIB_FRAMES = 60  # webcam calibration finishes by itself after this many detected frames


def commit_calibration(db, uid, pose: str, variant: str, prof: dict) -> None:
    """Save the profile and move straight on to Training."""
    db.save_calibration(uid, pose, variant, prof)
    base = criteria_for(pose, variant)
    changed = describe_adaptation(base, apply_calibration(base, prof))
    st.session_state.calib_notice = (
        f"Calibration saved from {prof['n_samples']} frames. "
        + (f"{len(changed)} target range(s) were adapted to your body." if changed
           else "Your values were already inside the default ranges."))
    st.session_state.pending_profile = None
    goto_now("Training")


def page_calibration(db, uid):
    stepper("Calibration")
    pose = st.session_state.pose
    spec = get_pose(pose)
    st.header(f"Calibrate: {pose}")
    saved = db.get_calibrations(uid, pose)

    if saved:
        st.success(f"You already have a saved calibration for {pose}.")
        c1, c2 = st.columns([1, 1])
        c1.button("Continue to training  →", type="primary", on_click=goto, args=("Training",), key="calib_continue")
        c2.button("← Change pose", on_click=goto, args=("Pose Selection",), key="calib_back")
        with st.expander("View / delete saved calibration", expanded=False):
            for v, prof in saved.items():
                st.markdown(f"**{spec.variant_labels.get(v, v)}** - {prof['n_samples']} samples, {prof['created_at']}")
                rows = describe_adaptation(criteria_for(pose, v), apply_calibration(criteria_for(pose, v), prof))
                st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame({"Info": ["No ranges needed changing."]}), hide_index=True)
            if st.button("Delete calibration for this pose"):
                db.delete_calibrations(uid, pose)
                st.rerun()
        st.subheader("Recalibrate (optional)")
    else:
        c1, c2 = st.columns([1, 1])
        c1.button("Skip for now  →", on_click=goto, args=("Training",), key="calib_skip")
        c2.button("← Change pose", on_click=goto, args=("Pose Selection",), key="calib_back2")

    st.markdown(
        '<div class="tip"><b>How it works:</b> 1) start the camera, 2) step back until your whole body is visible, '
        '3) hold your most comfortable version of the pose. Recording stops by itself after about 5 seconds and you '
        'are taken to Training. Ranges are only ever widened (by a limited amount) - never tightened.</div>',
        unsafe_allow_html=True)
    variant = calibration_variant_choice(pose)
    st.info(f"\U0001F4F7 {spec.camera_hint}")
    st.caption("Calibrate with a genuine attempt at the pose. It personalises the *limits* of the ranges - it does not "
               "certify that a position is safe for your body.")

    tabs = st.tabs(["Webcam", "Upload video", "Photo"])
    params = {"pose": pose, "variant": variant, "level": "Intermediate", "calibrations": None,
              "mirror": st.session_state.mirror, "model_variant": st.session_state.model_variant, "mode": "calibrate"}

    with tabs[0]:
        if not WEBRTC_AVAILABLE:
            st.warning("`streamlit-webrtc` is not installed - use the video or photo tab.")
        else:
            ctx = webrtc_streamer(key=f"calib-{pose}-{variant}", mode=WebRtcMode.SENDRECV,
                                  rtc_configuration={"iceServers": get_ice_servers()},
                                  media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                                  video_processor_factory=YogaProcessor, async_processing=True, **webrtc_extra())
            if ctx.video_processor:
                ctx.video_processor.set_params(params)
            c1, c2 = st.columns(2)
            finish = c1.button("Finish now", key="calib_finish")
            if c2.button("Restart recording", key="calib_restart") and ctx.video_processor:
                ctx.video_processor.restart()
            info = st.empty()

            def calib_profile():
                proc = ctx.video_processor
                if proc and proc.pipeline and proc.pipeline.calibrator:
                    return proc.pipeline.calibrator.build_profile()
                return None

            if finish:
                try:
                    prof = calib_profile()
                    if prof is not None:
                        commit_calibration(db, uid, pose, variant, prof)
                    else:
                        st.warning("Start the camera and hold the pose first.")
                except ValueError as exc:
                    st.error(str(exc))
            elif ctx.state.playing and ctx.video_processor:
                while ctx.state.playing:
                    pipe = ctx.video_processor.pipeline
                    n = pipe.calibrator.n_samples if pipe and pipe.calibrator else 0
                    info.progress(min(1.0, n / CALIB_FRAMES),
                                  text=f"Recording... {min(n, CALIB_FRAMES)}/{CALIB_FRAMES} frames - hold still in your pose"
                                  if n else "Waiting for your body to be detected - step back so you are fully in view")
                    if n >= CALIB_FRAMES:
                        try:
                            prof = calib_profile()
                        except ValueError:
                            prof = None
                        if prof is not None:
                            info.success("Calibration complete - moving on to training...")
                            time.sleep(0.8)
                            commit_calibration(db, uid, pose, variant, prof)
                    time.sleep(0.4)
            else:
                info.caption("The camera starts automatically (allow access in your browser). "
                             "Press **Start** if it does not.")

    with tabs[1]:
        vid = st.file_uploader("Video of you holding the pose", type=["mp4", "mov", "avi", "mkv", "webm"], key="calib_video")
        if vid is not None and st.button("Calibrate from video", type="primary"):
            pipe = YogaPipeline(make_detector(), pose, variant, "Intermediate", None, source="video")
            pipe.start_calibration()
            analyze_video(vid, pipe, max_seconds=30)
            try:
                commit_calibration(db, uid, pose, variant, pipe.calibrator.build_profile())
            except ValueError as exc:
                st.error(str(exc))

    with tabs[2]:
        shot = st.camera_input("Take a photo of your comfortable pose", key="calib_photo")
        if shot is not None and st.button("Calibrate from photo", type="primary"):
            img = decode_image(shot.getvalue())
            pipe = YogaPipeline(make_detector(), pose, variant, "Intermediate", None, source="photo")
            pipe.start_calibration()
            res = pipe.process_frame(img, 0.0)
            if res.detected:
                st.caption("A single photo gives only one sample - a short video or webcam recording is more reliable.")
                commit_calibration(db, uid, pose, variant, pipe.calibrator.build_profile())
            else:
                st.error(res.message)


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
                st.markdown(f'<div class="corr">{corr.text}</div>', unsafe_allow_html=True)
        elif res.detected and res.is_correct:
            st.markdown('<div class="good">Great posture - keep holding steady.</div>', unsafe_allow_html=True)
        else:
            st.write(res.message)


def finalize_session(db, uid, summary: dict, frames=None) -> bool:
    """Save a finished session. Returns True if saved (the caller then moves on to the results page)."""
    pose = summary["pose"]
    if summary["detected_frames"] < 10:
        st.error("Too few frames with a detected body - nothing was saved. Check lighting and make sure your whole body is visible.")
        return False
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
    return True


def page_training(db, uid):
    stepper("Training")
    pose, variant = st.session_state.pose, st.session_state.variant
    spec = get_pose(pose)
    level = current_level(db, uid, pose)
    cfg = LEVELS[level]
    st.header("Training")
    notice = st.session_state.calib_notice
    if notice:
        st.success("✅ " + notice)
        st.session_state.calib_notice = None
    m = st.columns(4)
    m[0].metric("Pose", pose)
    m[1].metric("Side", {"auto": "Auto-detect"}.get(variant, spec.variant_labels.get(variant, variant)))
    m[2].metric("Difficulty", level)
    m[3].metric("Hold target", f"{cfg.hold_target_s:.0f}s" + (" (one go)" if cfg.strict_hold else ""))
    st.markdown(f'<div class="tip">\U0001F4F7 {spec.camera_hint} Keep your whole body in view and well lit. '
                + ("Your session ends and results open automatically when you reach the hold target."
                   if st.session_state.auto_finish else "Press Finish when you are done.") + '</div>',
                unsafe_allow_html=True)
    if db.get_calibrations(uid, pose):
        st.caption("✅ Personal calibration is active for this pose.")
    else:
        st.caption("No calibration yet - default ranges are used.")
        st.button("Calibrate first", on_click=goto, args=("Calibration",), key="train_to_calib")

    tabs = st.tabs(["Live webcam", "Upload video", "Photo snapshot", "Demo (no camera)"])

    with tabs[0]:
        if not WEBRTC_AVAILABLE:
            st.warning("`streamlit-webrtc` is not installed - use the video upload tab.")
        else:
            video_col, panel_col = st.columns([3, 2], gap="large")
            with video_col:
                ctx = webrtc_streamer(key=f"train-{pose}", mode=WebRtcMode.SENDRECV,
                                      rtc_configuration={"iceServers": get_ice_servers()},
                                      media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                                      video_processor_factory=YogaProcessor, async_processing=True, **webrtc_extra())
                b1, b2 = st.columns(2)
                finish = b1.button("Finish & see results", type="primary", key="train_finish")
                restart = b2.button("Restart session", key="train_restart")
            panel = panel_col.empty()
            if ctx.video_processor:
                ctx.video_processor.set_params(live_params(db, uid, "train"))
                if restart:
                    ctx.video_processor.restart()
            if finish:
                proc = ctx.video_processor
                if proc and proc.pipeline and proc.pipeline.recorder:
                    summary = proc.pipeline.recorder.summary()
                    saved_ok = finalize_session(db, uid, summary)
                    proc.restart()
                    if saved_ok:
                        goto_now("Session Results")
                else:
                    st.warning("Start the camera first, practise the pose, then press Finish.")
            elif ctx.state.playing and ctx.video_processor:
                proc = ctx.video_processor
                while ctx.state.playing:
                    if proc.error:
                        panel.error(proc.error)
                    else:
                        res = proc.get_latest()
                        render_live(panel, res)
                        reached = (res is not None and res.detected
                                   and float((res.hold or {}).get("progress", 0.0)) >= 1.0)
                        if reached and st.session_state.auto_finish and proc.pipeline and proc.pipeline.recorder:
                            summary = proc.pipeline.recorder.summary()
                            panel.success("\U0001F389 Hold target reached - saving your session...")
                            time.sleep(1.2)
                            saved_ok = finalize_session(db, uid, summary)
                            proc.restart()
                            if saved_ok:
                                goto_now("Session Results")
                            break
                    time.sleep(0.3)
            else:
                panel.info("Waiting for the camera... allow access in your browser. If the video never appears, your "
                           "network may block WebRTC - use the video upload or photo tab.")

    with tabs[1]:
        vid = st.file_uploader("Upload a video of your practice (MP4 recommended)", type=["mp4", "mov", "avi", "mkv", "webm"], key="train_video")
        max_s = st.slider("Analyse at most (seconds)", 10, 180, 90, key="train_max_s")
        if vid is not None and st.button("Analyse video", type="primary", key="train_video_go"):
            pipe = build_pipeline(db, uid, "video")
            best, worst = analyze_video(vid, pipe, float(max_s))
            if finalize_session(db, uid, pipe.recorder.summary(), {"best": best, "worst": worst}):
                goto_now("Session Results")

    with tabs[2]:
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
                if finalize_session(db, uid, summary):
                    goto_now("Session Results")
            else:
                st.session_state.last_summary = summary
                st.session_state.last_frames = None
                st.session_state.last_notes = session_feedback(summary)
                goto_now("Session Results")


def page_results(db, uid):
    stepper("Session Results")
    st.header("Session results")
    sessions = db.get_sessions(uid)
    options = ["Latest session (this visit)"] if st.session_state.last_summary else []
    ids = sessions["id"].tolist()[::-1] if len(sessions) else []
    labels = {i: f"#{i} - {sessions.loc[sessions['id'] == i, 'pose'].iloc[0]} - "
                 f"{sessions.loc[sessions['id'] == i, 'started_at'].iloc[0][:16].replace('T', ' ')}" for i in ids}
    if not options and not ids:
        st.info("No sessions yet. Complete a training session first.")
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
    if frames and (frames.get("best") is not None):
        f1, f2 = st.columns(2)
        f1.image(frames["best"], caption="Best moment")
        f2.image(frames["worst"], caption="Needs most work")
    st.divider()
    n1, n2, n3 = st.columns(3)
    n1.button("Practise again  ↺", type="primary", on_click=goto, args=("Training",), key="res_again")
    n2.button("Try another pose", on_click=goto, args=("Pose Selection",), key="res_other")
    n3.button("See my progress", on_click=goto, args=("Progress Dashboard",), key="res_progress")


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
            f'<br><i>Tip: {r["tip"]}</i></div>', unsafe_allow_html=True)


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
    queued = st.session_state.get("_goto")
    if queued:                      # auto-advance requested by the previous run
        st.session_state.page = queued
        st.session_state._goto = None
    st.markdown(CSS, unsafe_allow_html=True)
    db = get_db()
    uid = sidebar(db)
    page = st.session_state.page
    {"Home": page_home, "Pose Selection": page_pose_selection, "Calibration": page_calibration,
     "Training": page_training, "Session Results": page_results, "Progress Dashboard": page_progress,
     "Recurring Errors": page_recurring, "Recommendations": page_recommendations}[page](db, uid)


main()
