
from modules.history_store import (
    init_database,
    save_session,
    get_sessions,
    get_session,
    delete_session,
    clear_history,
)

init_database()

import os
import re
import html
import tempfile
import hashlib

import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.audio_processing import analyze_audio
from modules.feature_extraction import extract_features
from modules.dtw_alignment import align_features
from modules.flaw_detection import detect_flaws, detect_individual_flaws
from modules.forced_alignment import forced_align
from modules.transcription import transcribe_audio
from modules.speech_coach import generate_speech_coaching


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SpeakWise AI | Speech Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PREMIUM DASHBOARD THEME
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #080d19;
        --panel: #111a2b;
        --panel-soft: #17243a;
        --border: rgba(148, 163, 184, .15);
        --text: #f1f5ff;
        --muted: #94a3b8;
        --blue: #60a5fa;
        --purple: #a78bfa;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(59,130,246,.13), transparent 35%),
            radial-gradient(ellipse at 95% 5%, rgba(139,92,246,.11), transparent 30%),
            #080d19;
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(8,13,25,.86);
        backdrop-filter: blur(14px);
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1500px;
        padding: 2rem 3rem 4rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #f1f5ff !important;
        letter-spacing: -.035em;
    }

    h1 {
        font-size: 2.55rem !important;
        font-weight: 700 !important;
    }

    h2 {
        font-size: 1.65rem !important;
        font-weight: 650 !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }

    p, label, li, .stMarkdown {
        color: #c6d1e5;
    }

    [data-testid="stCaptionContainer"] {
        color: #94a3b8;
        font-size: .84rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111b2d, #0b1220);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #edf2ff;
    }

    [data-testid="stSidebar"] [role="radio"] {
        background: rgba(255,255,255,.025);
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 5px;
    }

    [data-testid="stSidebar"] [role="radio"]:hover {
        background: rgba(96,165,250,.10);
        border-color: rgba(96,165,250,.2);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #17253c, #101a2c);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 19px 20px;
        min-height: 110px;
        box-shadow: 0 8px 25px rgba(0,0,0,.15);
    }

    [data-testid="stMetricLabel"] {
        color: #9eacc5 !important;
        font-size: .82rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f4f7ff !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Upload */
    [data-testid="stFileUploader"] {
        background: rgba(20,31,52,.65);
        border: 1px dashed rgba(110,168,254,.4);
        border-radius: 15px;
        padding: 14px;
        transition: .2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #60a5fa;
        background: rgba(30,48,78,.7);
    }

    [data-testid="stAudio"] {
        background: #111c30;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 8px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(110,168,254,.35);
        padding: .62rem 1.2rem;
        font-weight: 650;
        transition: all .2s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #3478e5, #7657d5);
        color: white;
        border: none;
        box-shadow: 0 6px 20px rgba(74,112,220,.22);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        border-color: #8bb7ff;
        box-shadow: 0 8px 22px rgba(70,115,210,.2);
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(17,26,43,.85);
        border: 1px solid var(--border);
        border-radius: 13px;
        margin-bottom: 10px;
    }

    [data-testid="stExpander"] summary {
        color: #e8efff;
        font-weight: 600;
    }

    /* Tabs */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 10px;
        border-bottom: 1px solid var(--border);
    }

    [data-testid="stTabs"] button[role="tab"] {
        color: #aebbd2;
        font-weight: 600;
        padding: 12px 16px;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #a9caff;
        border-bottom: 2px solid #60a5fa;
    }

    /* Tables and charts */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stPlotlyChart"] {
        background: rgba(17,26,43,.72);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 8px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Inputs */
    input, textarea, [data-baseweb="select"] > div {
        background-color: #111c30 !important;
        border-color: rgba(148,163,184,.22) !important;
        border-radius: 9px !important;
        color: #edf2ff !important;
    }

    hr {
        border-color: rgba(148,163,184,.13);
        margin: 1.7rem 0;
    }

    .sw-hero {
        background: linear-gradient(115deg, rgba(32,67,120,.42), rgba(79,49,130,.30));
        border: 1px solid rgba(125,155,220,.22);
        border-radius: 20px;
        padding: 25px 28px;
        margin: 8px 0 24px 0;
    }

    .sw-hero-title {
        color: #f4f7ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .sw-hero-subtitle {
        color: #b8c7df;
        font-size: 14px;
        line-height: 1.6;
    }

    .sw-section-label {
        color: #8fbaff;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .sw-flaw-card {
        background: linear-gradient(145deg, #17243a, #101a2b);
        border: 1px solid rgba(148,163,184,.16);
        border-radius: 14px;
        padding: 16px 18px;
        margin: 8px 0;
    }

    .sw-flaw-title {
        color: #f1f5ff;
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 7px;
    }

    .sw-flaw-meta {
        color: #a9b8d0;
        font-size: 12px;
        line-height: 1.7;
    }

    .sw-footer {
        border-top: 1px solid rgba(148,163,184,.13);
        margin-top: 35px;
        padding-top: 16px;
        color: #71819a;
        font-size: 12px;
        text-align: center;
    }

    @media (max-width: 900px) {
        [data-testid="stAppViewContainer"] .main .block-container {
            padding: 1.2rem 1rem 3rem;
        }

        h1 {
            font-size: 2rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PLOTLY DEFAULT THEME
# =========================================================

PLOT_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {
        "family": "DM Sans, sans-serif",
        "color": "#c6d1e5",
    },
    "margin": {"l": 45, "r": 25, "t": 55, "b": 45},
    "hovermode": "x unified",
}


def style_chart(fig, height=360):
    fig.update_layout(
        **PLOT_LAYOUT,
        height=height,
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "#c6d1e5"},
        },
    )
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,.10)",
        zerolinecolor="rgba(148,163,184,.12)",
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,.10)",
        zerolinecolor="rgba(148,163,184,.12)",
    )
    return fig


# =========================================================
# SESSION STATE
# =========================================================

def init_state():
    defaults = {
        "individual_audio_path": None,
        "individual_audio_signature": None,
        "individual_results": None,
        "individual_features": None,
        "individual_flaws": [],
        "alignment_signature": None,
        "alignment_transcript": None,
        "alignment_words": None,
        "alignment_audio_path": None,
        "alignment_error": None,
        "comparison_word_mapping": None,
        "last_analyzed_signature": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_uploaded_audio(uploaded_file, temp_paths=None):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        path = temp_file.name

    if temp_paths is not None:
        temp_paths.append(path)

    return path


def get_upload_signature(uploaded_file):
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def uploaded_files_are_identical(file_a, file_b):
    if file_a is None or file_b is None:
        return False

    return get_upload_signature(file_a) == get_upload_signature(file_b)


def get_file_signature(path):
    digest = hashlib.sha256()

    with open(path, "rb") as audio_file:
        for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_persistent_audio(uploaded_file, state_key):
    signature = get_upload_signature(uploaded_file)
    signature_key = f"{state_key}_signature"
    path_key = f"{state_key}_path"

    existing_path = st.session_state.get(path_key)
    existing_signature = st.session_state.get(signature_key)

    if (
        existing_signature == signature
        and existing_path
        and os.path.exists(existing_path)
    ):
        return existing_path

    if existing_path and os.path.exists(existing_path):
        try:
            os.remove(existing_path)
        except OSError:
            pass

    new_path = save_uploaded_audio(uploaded_file)

    st.session_state[path_key] = new_path
    st.session_state[signature_key] = signature

    return new_path


@st.cache_data(show_spinner=False)
def load_audio_for_playback(audio_path):
    return librosa.load(audio_path, sr=16000, mono=True)


init_state()


# =========================================================
# COMMON UI HELPERS
# =========================================================

def show_hero(title, subtitle, eyebrow="SPEECH INTELLIGENCE"):
    st.markdown(
        f"""
        <div class="sw-hero">
            <div class="sw-section-label">{html.escape(eyebrow)}</div>
            <div class="sw-hero-title">{html.escape(title)}</div>
            <div class="sw-hero-subtitle">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_footer():
    st.markdown(
        """
        <div class="sw-footer">
            SpeakWise AI &nbsp;·&nbsp; Contrastive Speech Intelligence
            &nbsp;·&nbsp; Preliminary Research Prototype
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_color(severity):
    try:
        severity = int(severity)
    except (TypeError, ValueError):
        return "#94a3b8"

    if severity >= 5:
        return "#f87171"
    if severity == 4:
        return "#fb923c"
    if severity == 3:
        return "#fbbf24"
    if severity == 2:
        return "#60a5fa"
    return "#34d399"


def severity_label(severity):
    try:
        severity = int(severity)
    except (TypeError, ValueError):
        return "Unrated"

    return {
        1: "Very Low",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Very High",
    }.get(severity, "Unrated")


def show_metric_row(items):
    columns = st.columns(len(items))

    for column, item in zip(columns, items):
        column.metric(item["label"], item["value"])


# =========================================================
# ACOUSTIC MEASUREMENT HELPERS
# =========================================================

def get_pitch_measurements(features):
    pitch = np.asarray(features.get("pitch", []), dtype=float)
    pitch = pitch[np.isfinite(pitch)]
    pitch = pitch[(pitch >= 65) & (pitch <= 400)]

    if len(pitch) < 2:
        return {
            "valid_frames": len(pitch),
            "median_hz": None,
            "pitch_range_hz": None,
            "pitch_std_hz": None,
        }

    return {
        "valid_frames": len(pitch),
        "median_hz": round(float(np.median(pitch)), 2),
        "pitch_range_hz": round(
            float(np.percentile(pitch, 90) - np.percentile(pitch, 10)),
            2,
        ),
        "pitch_std_hz": round(float(np.std(pitch)), 2),
    }


def get_activity_measurements(features):
    activity = np.asarray(
        features.get("speech_activity", []),
        dtype=float,
    )
    activity = activity[np.isfinite(activity)]

    if len(activity) == 0:
        return {
            "mean_activity": None,
            "low_activity_windows": 0,
            "window_count": 0,
        }

    return {
        "mean_activity": round(float(np.mean(activity)), 3),
        "low_activity_windows": int(np.sum(activity < 0.10)),
        "window_count": len(activity),
    }


# =========================================================
# DELIVERY FLAW RESULTS
# =========================================================

def show_flaw_results(flaws, recording_duration=None):
    st.subheader("🔎 Detected Delivery Flaws")

    if not flaws:
        st.success(
            "No potential delivery flaws were flagged by the current rules."
        )
        st.caption(
            "This does not guarantee flawless speech. Always review the recording."
        )
        return

    ranked_flaws = sorted(
        flaws,
        key=lambda item: (
            -int(item.get("severity", 0) or 0),
            float(item.get("start", 0) or 0),
        ),
    )

    total_flaws = len(ranked_flaws)
    high_priority = sum(
        1 for item in ranked_flaws
        if int(item.get("severity", 0) or 0) >= 4
    )
    moderate_priority = sum(
        1 for item in ranked_flaws
        if int(item.get("severity", 0) or 0) == 3
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("Potential Indicators", total_flaws)
    metric_cols[1].metric("High Priority", high_priority)
    metric_cols[2].metric("Moderate Priority", moderate_priority)

    st.markdown("#### Severity-ranked summary")

    for index, flaw in enumerate(ranked_flaws, start=1):
        severity = flaw.get("severity", "N/A")
        color = severity_color(severity)
        name = html.escape(str(flaw.get("flaw", "Potential flaw")))
        level = html.escape(
            str(flaw.get("severity_label", severity_label(severity)))
        )

        start = flaw.get("start")
        end = flaw.get("end")

        if start is not None and end is not None:
            timestamp = f"{float(start):.2f}s – {float(end):.2f}s"
        else:
            timestamp = "Timestamp unavailable"

        duration = flaw.get("duration")
        if duration is None and start is not None and end is not None:
            duration = max(0, float(end) - float(start))

        duration_text = (
            f"{float(duration):.2f} sec"
            if duration is not None
            else "N/A"
        )

        st.markdown(
            f"""
            <div class="sw-flaw-card">
                <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;">
                    <div class="sw-flaw-title">{index}. {name}</div>
                    <div style="color:{color};font-weight:800;font-size:12px;">
                        {html.escape(str(severity))}/5 · {level}
                    </div>
                </div>
                <div class="sw-flaw-meta">
                    <span>◷ {timestamp}</span>
                    &nbsp;&nbsp;·&nbsp;&nbsp;
                    <span>Duration: {duration_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    summary_rows = []
    evidence_rows = []

    for flaw in ranked_flaws:
        start = flaw.get("start")
        end = flaw.get("end")
        duration = flaw.get("duration")

        if duration is None and start is not None and end is not None:
            duration = max(0.0, float(end) - float(start))

        severity = flaw.get("severity", "N/A")

        summary_rows.append({
            "Severity": f"{severity}/5",
            "Level": severity_label(severity),
            "Potential indicator": flaw.get("flaw", "Potential flaw"),
            "Start (s)": round(float(start), 2) if start is not None else None,
            "End (s)": round(float(end), 2) if end is not None else None,
            "Duration (s)": round(float(duration), 2) if duration is not None else None,
        })

        evidence_rows.append({
            "Potential indicator": flaw.get("flaw", "Potential flaw"),
            "Severity": f"{severity}/5",
            "Time (s)": (
                f"{float(start):.2f}–{float(end):.2f}"
                if start is not None and end is not None
                else "Unavailable"
            ),
            "Evidence": flaw.get("evidence", "Not available"),
            "Interpretation": flaw.get("explanation", "Not available"),
        })

    with st.expander("View complete severity summary table"):
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Severity is a preliminary rule-based priority, not a validated "
        "pronunciation or presentation-quality score."
    )

    intervals = [
        flaw for flaw in ranked_flaws
        if flaw.get("start") is not None
        and flaw.get("end") is not None
    ]

    if intervals:
        st.markdown("#### ⏱️ Potential flaw timeline")

        timeline = go.Figure()

        for flaw in sorted(intervals, key=lambda item: float(item["start"])):
            start = max(0.0, float(flaw["start"]))
            end = max(start, float(flaw["end"]))
            duration = max(0.05, end - start)
            severity = flaw.get("severity", "N/A")
            color = severity_color(severity)
            label = (
                f"{flaw.get('flaw', 'Potential flaw')} "
                f"({severity}/5)"
            )

            timeline.add_trace(
                go.Bar(
                    x=[duration],
                    y=[label],
                    base=[start],
                    orientation="h",
                    marker_color=color,
                    showlegend=False,
                    customdata=[[start, end, severity]],
                    hovertemplate=(
                        "%{y}<br>Start: %{customdata[0]:.2f} s"
                        "<br>End: %{customdata[1]:.2f} s"
                        "<br>Severity: %{customdata[2]}/5<extra></extra>"
                    ),
                )
            )

        axis_end = max(
            float(recording_duration or 0),
            max(float(item["end"]) for item in intervals),
            1.0,
        )

        timeline.update_layout(
            title="Potential Delivery Flaws Across Recording",
            xaxis_title="Participant recording time (seconds)",
            yaxis_title="Flagged indicator",
            xaxis={"range": [0, axis_end], "rangemode": "tozero"},
            barmode="overlay",
            height=max(300, 55 * len(intervals)),
            margin={"l": 20, "r": 25, "t": 55, "b": 50},
            hovermode="closest",
        )

        style_chart(timeline, max(300, 55 * len(intervals)))
        st.plotly_chart(timeline, use_container_width=True)

    st.markdown("#### 📋 Evidence and explanations")

    st.dataframe(
        pd.DataFrame(evidence_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Evidence": st.column_config.TextColumn(
                "Evidence",
                width="medium",
            ),
            "Interpretation": st.column_config.TextColumn(
                "Interpretation",
                width="large",
            ),
        },
    )

    st.caption(
        "These are potential acoustic indicators. Listen to the original "
        "recording before treating them as actual delivery flaws."
    )


# =========================================================
# AI SPEECH IMPROVEMENT COACH
# =========================================================

def show_speech_coach(flaws):
    st.subheader("🧠 AI Speech Improvement Coach")

    if not flaws:
        st.info(
            "No potential flaws were flagged. You can still practise "
            "clear, natural and confident delivery."
        )
        return

    coaching = generate_speech_coaching(flaws)

    if not coaching:
        st.info("No coaching suggestions are available.")
        return

    st.caption(
        "Practice suggestions based on detected acoustic indicators. "
        "Review the recording before acting on a suggestion."
    )

    st.markdown("#### Your priority practice plan")

    for index, item in enumerate(coaching, start=1):
        start = item.get("start")
        end = item.get("end")

        if start is not None and end is not None:
            timestamp = f"{float(start):.2f}–{float(end):.2f} sec"
        else:
            timestamp = "Timestamp unavailable"

        with st.expander(
            f"{index}. {item['title']} — {item['priority']} "
            f"(Severity {item['severity']}/5)",
            expanded=(index == 1),
        ):
            st.write(f"**Recording segment:** {timestamp}")
            st.write("**What was noticed**")
            st.write(item["observation"])
            st.write("**Acoustic evidence**")
            st.write(item["evidence"])
            st.write("**How to improve**")
            st.write(item["improvement_tip"])
            st.write("**Practice exercise**")
            st.write(item["practice_exercise"])
            st.caption(item["verification_note"])


# =========================================================
# WORD-LEVEL FLAW MAPPING
# =========================================================

def show_word_level_flaw_mapping(audio_path, flaws):
    st.subheader("🧩 Word-Level Flaw Mapping")

    if not flaws:
        st.info("No detected intervals need word-level mapping.")
        return

    if not audio_path or not os.path.exists(audio_path):
        st.error("The participant audio file is unavailable.")
        return

    signature_data = (
        get_file_signature(audio_path),
        tuple(
            (
                item.get("flaw"),
                item.get("start"),
                item.get("end"),
            )
            for item in flaws
        ),
    )

    mapping_signature = hashlib.sha256(
        repr(signature_data).encode("utf-8")
    ).hexdigest()

    cached = st.session_state.get("comparison_word_mapping")

    if not cached or cached.get("signature") != mapping_signature:
        try:
            with st.spinner("Transcribing and aligning words once..."):
                transcript = transcribe_audio(audio_path)

                if not transcript:
                    st.warning("No transcript was generated.")
                    return

                if re.search(r"[\u0900-\u097F]", transcript):
                    st.warning(
                        "The transcript contains Devanagari Hindi. "
                        "The current forced-alignment module supports "
                        "English/Romanized text only."
                    )
                    st.text_area(
                        "Generated transcript",
                        value=transcript,
                        height=80,
                        key="mapping_hindi_transcript",
                    )
                    return

                words = forced_align(audio_path, transcript)

                st.session_state["comparison_word_mapping"] = {
                    "signature": mapping_signature,
                    "transcript": transcript,
                    "words": words or [],
                }

        except Exception as error:
            st.error(f"Word-level mapping failed: {error}")
            return

    cached = st.session_state["comparison_word_mapping"]
    valid_words = []

    for word in cached.get("words", []):
        try:
            start = float(word["start"])
            end = float(word["end"])
            text = str(word.get("word", "")).strip()

            if text and start >= 0 and end > start:
                valid_words.append({
                    "word": text,
                    "start": start,
                    "end": end,
                })
        except (KeyError, TypeError, ValueError):
            continue

    if not valid_words:
        st.warning("No valid word timestamps were returned.")
        return

    st.caption(
        f"{len(valid_words)} words aligned. A match means timestamp overlap; "
        "it does not prove that a particular word was mispronounced."
    )

    with st.expander("View generated transcript"):
        st.write(cached.get("transcript", ""))

    mapping_rows = []
    word_options = {}

    for flaw_index, flaw in enumerate(flaws):
        try:
            start = float(flaw["start"])
            end = float(flaw["end"])
        except (KeyError, TypeError, ValueError):
            continue

        if end <= start:
            continue

        matched = [
            word for word in valid_words
            if word["start"] < end and word["end"] > start
        ]

        flaw_name = flaw.get("flaw", "Potential flaw")
        severity = flaw.get("severity", "N/A")

        if not matched:
            mapping_rows.append({
                "Flaw": flaw_name,
                "Severity": f"{severity}/5",
                "Flaw interval (s)": f"{start:.2f}–{end:.2f}",
                "Overlapping word": "No aligned word",
                "Word interval (s)": "—",
            })
            continue

        for word in matched:
            label = (
                f"{word['word']} "
                f"({word['start']:.2f}–{word['end']:.2f} s)"
            )
            unique_key = f"Flaw {flaw_index + 1} | {label}"
            word_options[unique_key] = word

            mapping_rows.append({
                "Flaw": flaw_name,
                "Severity": f"{severity}/5",
                "Flaw interval (s)": f"{start:.2f}–{end:.2f}",
                "Overlapping word": word["word"],
                "Word interval (s)": (
                    f"{word['start']:.2f}–{word['end']:.2f}"
                ),
            })

    st.dataframe(
        pd.DataFrame(mapping_rows),
        use_container_width=True,
        hide_index=True,
    )

    if word_options:
        selected_label = st.selectbox(
            "Choose a mapped word to listen",
            list(word_options.keys()),
            key="comparison_mapped_word_audio",
        )

        selected = word_options[selected_label]

        try:
            audio, sr = load_audio_for_playback(audio_path)
            audio_duration = len(audio) / sr

            clip_start = max(0.0, selected["start"] - 0.15)
            clip_end = min(audio_duration, selected["end"] + 0.15)

            start_sample = int(clip_start * sr)
            end_sample = min(len(audio), int(clip_end * sr))

            if end_sample > start_sample:
                st.audio(
                    audio[start_sample:end_sample],
                    sample_rate=sr,
                )

        except Exception as error:
            st.warning(f"Could not play the selected word: {error}")


# =========================================================
# DIAGNOSTIC MEASUREMENTS
# =========================================================

def show_diagnostics(reference_features, participant_features):
    st.subheader("🧪 Diagnostic Measurements")

    st.caption(
        "These measurements explain current rules; they are not validated scores."
    )

    ref_pitch = get_pitch_measurements(reference_features)
    part_pitch = get_pitch_measurements(participant_features)

    ref_activity = get_activity_measurements(reference_features)
    part_activity = get_activity_measurements(participant_features)

    rows = [
        {
            "Measurement": "Valid pitch frames",
            "Reference": ref_pitch["valid_frames"],
            "Participant": part_pitch["valid_frames"],
        },
        {
            "Measurement": "Pitch range (90th–10th percentile, Hz)",
            "Reference": ref_pitch["pitch_range_hz"],
            "Participant": part_pitch["pitch_range_hz"],
        },
        {
            "Measurement": "Pitch standard deviation (Hz)",
            "Reference": ref_pitch["pitch_std_hz"],
            "Participant": part_pitch["pitch_std_hz"],
        },
        {
            "Measurement": "Median pitch (Hz)",
            "Reference": ref_pitch["median_hz"],
            "Participant": part_pitch["median_hz"],
        },
        {
            "Measurement": "Mean speech activity (0–1)",
            "Reference": ref_activity["mean_activity"],
            "Participant": part_activity["mean_activity"],
        },
        {
            "Measurement": "Low-activity windows (< 0.10)",
            "Reference": ref_activity["low_activity_windows"],
            "Participant": part_activity["low_activity_windows"],
        },
    ]

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    ref_range = ref_pitch["pitch_range_hz"]
    part_range = part_pitch["pitch_range_hz"]

    if ref_range is not None and part_range is not None and ref_range > 0:
        ratio = part_range / ref_range
        st.write(
            f"**Participant/reference pitch-range ratio:** {ratio:.2f}"
        )

        if ratio < 0.50:
            st.warning(
                "Participant pitch range is less than half the reference range. "
                "Listen before interpreting this difference."
            )

    ref_mean = ref_activity["mean_activity"]
    part_mean = part_activity["mean_activity"]

    if ref_mean is not None and part_mean is not None:
        st.write(
            f"**Mean speech-activity difference:** "
            f"{part_mean - ref_mean:+.3f}"
        )

    st.caption(
        "Low activity may indicate silence, but can also occur during quiet speech."
    )


# =========================================================
# WORD ALIGNMENT AND PLAYBACK
# =========================================================

def show_aligned_words(audio_path, word_results, key_prefix):
    if not word_results:
        st.warning("The alignment model returned no word timestamps.")
        return

    st.dataframe(
        pd.DataFrame(word_results),
        use_container_width=True,
        hide_index=True,
    )

    valid_words = [
        item for item in word_results
        if item.get("start") is not None
        and item.get("end") is not None
        and item["end"] > item["start"]
    ]

    if not valid_words:
        st.warning("No usable word boundaries were returned.")
        return

    st.markdown("#### 🔊 Listen to an aligned word")

    options = {
        f"{item['word']} ({item['start']:.2f}–{item['end']:.2f} sec)": item
        for item in valid_words
    }

    selected_label = st.selectbox(
        "Select a word to listen to",
        list(options.keys()),
        key=f"{key_prefix}_selected_word",
    )

    selected = options[selected_label]

    if not audio_path or not os.path.exists(audio_path):
        st.error("Saved audio is unavailable. Please upload it again.")
        return

    audio, sr = load_audio_for_playback(audio_path)

    start_sample = max(0, int(selected["start"] * sr))
    end_sample = min(len(audio), int(selected["end"] * sr))

    if end_sample > start_sample:
        st.audio(
            audio[start_sample:end_sample],
            sample_rate=sr,
        )

    st.caption(
        "Alignment confidence is not pronunciation quality. "
        "Verify word boundaries by listening."
    )


def run_auto_transcription_alignment(audio_path, key_prefix):
    signature = st.session_state.get("individual_audio_signature")

    if (
        st.session_state.get("alignment_signature") != signature
        or st.session_state.get("alignment_audio_path") != audio_path
    ):
        st.session_state["alignment_signature"] = signature
        st.session_state["alignment_transcript"] = None
        st.session_state["alignment_words"] = None
        st.session_state["alignment_audio_path"] = audio_path
        st.session_state["alignment_error"] = None

    if st.session_state["alignment_transcript"] is None:
        with st.spinner("Transcribing recording..."):
            try:
                st.session_state["alignment_transcript"] = (
                    transcribe_audio(audio_path)
                )
            except Exception as error:
                st.session_state["alignment_error"] = str(error)

    if st.session_state.get("alignment_error"):
        st.error(
            "Transcription failed: "
            + st.session_state["alignment_error"]
        )
        return

    transcript = st.session_state.get("alignment_transcript")

    if not transcript:
        st.warning("No transcript is available.")
        return

    st.subheader("📝 Automatically Generated Transcript")

    st.text_area(
        "Review the AI transcript",
        value=transcript,
        height=100,
        key=f"{key_prefix}_transcript_display",
    )

    if re.search(r"[\u0900-\u097F]", transcript):
        st.warning(
            "This transcript contains Devanagari Hindi. "
            "The current alignment module supports English/Romanized text only."
        )
        return

    if st.session_state.get("alignment_words") is None:
        with st.spinner("Aligning words with the recording..."):
            try:
                st.session_state["alignment_words"] = (
                    forced_align(audio_path, transcript)
                )
            except Exception as error:
                st.session_state["alignment_error"] = str(error)
                st.error(f"Word alignment failed: {error}")
                return

    words = st.session_state.get("alignment_words") or []

    if words:
        st.success("Transcript and word timestamps are ready.")
        show_aligned_words(
            st.session_state["alignment_audio_path"],
            words,
            key_prefix,
        )
    else:
        st.warning("No word timestamps were returned.")


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown(
    """
    <div style="padding:5px 0 18px 0;">
        <div style="font-size:25px;font-weight:800;color:#f1f5ff;">
            🎙️ SpeakWise AI
        </div>
        <div style="font-size:12px;color:#94a3b8;margin-top:5px;">
            Speech Intelligence Platform
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

st.sidebar.markdown(
    '<div class="sw-section-label">WORKSPACE</div>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Individual Speech Analysis",
        "🎯 Reference vs Participant",
        "📈 Progress Tracking",
    ],
    key="dashboard_navigation",
    label_visibility="collapsed",
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    <div style="padding:8px 2px;">
        <div style="font-size:11px;color:#60a5fa;font-weight:800;letter-spacing:1px;">
            POWERED BY AI
        </div>
        <div style="font-size:12px;color:#94a3b8;line-height:1.6;margin-top:6px;">
            Acoustic analysis<br>
            Temporal flaw grounding<br>
            Speech improvement insights
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PAGE 1 — INDIVIDUAL SPEECH ANALYSIS
# =========================================================

if page == "🏠 Individual Speech Analysis":

    show_hero(
        "Individual Speech Analysis",
        "Understand your speaking patterns through acoustic measurements, "
        "delivery indicators and actionable practice suggestions.",
        "PERSONAL SPEECH ANALYTICS",
    )

    upload_col, info_col = st.columns([1.5, 1])

    with upload_col:
        st.markdown("### 🎙️ Upload your recording")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "ogg"],
            key="individual_audio",
            help="Supported formats: WAV, MP3, M4A and OGG.",
        )

    with info_col:
        st.markdown("### What you will get")
        st.markdown(
            """
            - Acoustic overview and speech activity
            - Potential delivery flaws with timestamps
            - AI-generated practice suggestions
            - Automatic transcript and word timestamps
            """
        )

    if uploaded_file is not None:

        st.audio(uploaded_file)

        individual_audio_path = get_persistent_audio(
            uploaded_file,
            "individual_audio",
        )

        current_signature = get_upload_signature(uploaded_file)

        if (
            st.session_state.get("last_analyzed_signature")
            != current_signature
        ):
            st.session_state["individual_results"] = None
            st.session_state["individual_features"] = None
            st.session_state["individual_flaws"] = []
            st.session_state["last_analyzed_signature"] = current_signature

            st.session_state["alignment_signature"] = None
            st.session_state["alignment_transcript"] = None
            st.session_state["alignment_words"] = None
            st.session_state["alignment_audio_path"] = None
            st.session_state["alignment_error"] = None

        analysis_tab, alignment_tab = st.tabs([
            "📊 Acoustic Analysis",
            "📝 Transcription & Word Timestamps",
        ])

        with analysis_tab:

            st.markdown("### Analyze your delivery")

            if st.button(
                "🚀 Analyze Speech",
                type="primary",
                key="analyze_button",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Analyzing speech features and detecting delivery indicators..."
                    ):
                        results = analyze_audio(individual_audio_path)
                        features = extract_features(individual_audio_path)
                        individual_flaws = detect_individual_flaws(features)

                    st.session_state["individual_flaws"] = individual_flaws
                    st.session_state["individual_results"] = results
                    st.session_state["individual_features"] = features

                    individual_signature = get_upload_signature(uploaded_file)

                    individual_metrics = {
                        "duration": float(results["duration"]),
                        "speech_seconds": float(
                            results["estimated_speech_seconds"]
                        ),
                        "pause_count": int(results["pause_count"]),
                        "dtw_distance": None,
                        "flaw_count": len(individual_flaws),
                    }

                    individual_details = {
                        "filename": uploaded_file.name,
                        "duration": individual_metrics["duration"],
                        "estimated_speech_seconds": (
                            individual_metrics["speech_seconds"]
                        ),
                        "pause_count": individual_metrics["pause_count"],
                        "average_energy_db": results.get("average_energy_db"),
                        "pitch_median_hz": results.get("pitch_median_hz"),
                        "flaws": individual_flaws,
                    }

                    save_session(
                        session_type="individual",
                        title=f"Individual Analysis: {uploaded_file.name}",
                        metrics=individual_metrics,
                        details=individual_details,
                        signature=f"individual:{individual_signature}",
                    )

                    st.success(
                        "Speech analysis and flaw detection completed!"
                    )

                except Exception as error:
                    st.error(f"Analysis failed: {error}")
                    st.exception(error)

            results = st.session_state.get("individual_results")
            features = st.session_state.get("individual_features")

            if results is not None and features is not None:

                st.divider()
                st.success("Your acoustic analysis is ready.")

                individual_flaws = st.session_state.get(
                    "individual_flaws",
                    [],
                )

                show_flaw_results(
                    individual_flaws,
                    recording_duration=features.get("duration"),
                )

                show_speech_coach(individual_flaws)

                st.divider()
                st.subheader("📊 Acoustic Overview")

                metric_items = [
                    {
                        "label": "Recording Duration",
                        "value": f"{results['duration']} sec",
                    },
                    {
                        "label": "Estimated Speech",
                        "value": f"{results['estimated_speech_seconds']} sec",
                    },
                    {
                        "label": "Pause Count",
                        "value": results["pause_count"],
                    },
                ]

                show_metric_row(metric_items)

                col1, col2 = st.columns(2)

                col1.metric(
                    "Average Energy",
                    f"{results['average_energy_db']} dB",
                )

                pitch = results["pitch_median_hz"]

                col2.metric(
                    "Median Pitch",
                    f"{pitch} Hz" if pitch is not None else "Unavailable",
                )

                st.divider()
                st.subheader("📈 Acoustic Visualizations")

                pitch_fig = go.Figure()

                pitch_fig.add_trace(
                    go.Scatter(
                        x=features["pitch_times"],
                        y=features["pitch"],
                        mode="lines",
                        name="Estimated pitch",
                        line={"color": "#60a5fa", "width": 2},
                    )
                )

                pitch_fig.update_layout(
                    title="Pitch Variation Over Time",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Pitch (Hz)",
                )

                style_chart(pitch_fig)
                st.plotly_chart(
                    pitch_fig,
                    use_container_width=True,
                )

                energy_fig = go.Figure()

                energy_fig.add_trace(
                    go.Scatter(
                        x=features["times"],
                        y=features["energy"],
                        mode="lines",
                        name="RMS energy",
                        line={"color": "#a78bfa", "width": 2},
                    )
                )

                energy_fig.update_layout(
                    title="Energy Over Time",
                    xaxis_title="Time (seconds)",
                    yaxis_title="RMS energy",
                )

                style_chart(energy_fig)
                st.plotly_chart(
                    energy_fig,
                    use_container_width=True,
                )

                pace_fig = go.Figure()

                pace_fig.add_trace(
                    go.Bar(
                        x=features["pace_times"],
                        y=features["speech_activity"],
                        name="Speech activity",
                        marker_color="#34d399",
                    )
                )

                pace_fig.update_layout(
                    title="Speech Activity Over Time",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Speech activity ratio",
                    yaxis={"range": [0, 1]},
                )

                style_chart(pace_fig)
                st.plotly_chart(
                    pace_fig,
                    use_container_width=True,
                )

                st.subheader("⏸️ Detected Pause Intervals")

                pauses = results.get("pauses", [])

                if pauses:
                    st.dataframe(
                        pd.DataFrame(pauses),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info(
                        "No pauses of 0.25 seconds or longer were detected."
                    )

                st.caption(
                    "Acoustic measurements are indicators, not definitive "
                    "judgments of speech quality."
                )

        with alignment_tab:

            st.markdown("### Automatic transcription")
            st.write(
                "Generate a transcript and align individual words with "
                "their timestamps in the recording."
            )

            if st.button(
                "🤖 Auto Transcribe & Generate Timestamps",
                type="primary",
                key="individual_alignment_button",
            ):
                st.session_state["alignment_signature"] = None
                st.session_state["alignment_transcript"] = None
                st.session_state["alignment_words"] = None
                st.session_state["alignment_audio_path"] = (
                    individual_audio_path
                )
                st.session_state["alignment_error"] = None

            if (
                st.session_state.get("alignment_transcript") is not None
                or st.session_state.get("alignment_error") is not None
                or st.session_state.get("alignment_signature") is None
            ):
                if (
                    st.session_state.get("alignment_audio_path")
                    == individual_audio_path
                    and st.session_state.get("alignment_signature") is None
                ):
                    st.session_state["alignment_signature"] = (
                        st.session_state.get("individual_audio_signature")
                    )

                if (
                    st.session_state.get("alignment_audio_path")
                    == individual_audio_path
                ):
                    run_auto_transcription_alignment(
                        individual_audio_path,
                        "individual_auto_alignment",
                    )

    else:
        st.markdown(
            """
            <div class="sw-flaw-card" style="text-align:center;padding:35px;">
                <div style="font-size:35px;">🎧</div>
                <div class="sw-flaw-title" style="font-size:17px;">
                    Your analysis workspace is ready
                </div>
                <div class="sw-flaw-meta">
                    Upload a recording to explore your speech patterns
                    and receive improvement insights.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    show_footer()


# =========================================================
# PAGE 2 — REFERENCE VS PARTICIPANT COMPARISON
# =========================================================

if page == "🎯 Reference vs Participant":

    show_hero(
        "Reference vs Participant",
        "Compare two recordings of the same speech passage. "
        "Explore acoustic alignment, potential delivery differences "
        "and timestamp-grounded evidence.",
        "CONTRASTIVE SPEECH ANALYTICS",
    )

    st.markdown("### Upload recordings")

    reference_col, participant_col = st.columns(2)

    with reference_col:
        st.markdown("#### 🎯 Reference Speech")
        st.caption("Upload the reference or ideal delivery recording.")

        reference_file = st.file_uploader(
            "Choose reference audio",
            type=["wav", "mp3", "m4a", "ogg"],
            key="reference_audio",
        )

        if reference_file is not None:
            st.audio(reference_file)

    with participant_col:
        st.markdown("#### 🎙️ Participant Speech")
        st.caption("Upload the recording you want to evaluate.")

        participant_file = st.file_uploader(
            "Choose participant audio",
            type=["wav", "mp3", "m4a", "ogg"],
            key="participant_audio",
        )

        if participant_file is not None:
            st.audio(participant_file)

    same_recording = uploaded_files_are_identical(
        reference_file,
        participant_file,
    )

    if same_recording:
        st.warning(
            "Identical recordings detected. Upload a different participant "
            "recording to continue."
        )

    ready_to_compare = (
        reference_file is not None
        and participant_file is not None
        and not same_recording
    )

    st.divider()

    if ready_to_compare:
        st.success("Both recordings are ready for comparison.")

    if st.button(
        "🚀 Compare Recordings",
        type="primary",
        key="compare_button",
        disabled=not ready_to_compare,
        use_container_width=True,
    ):

        temp_paths = []

        try:
            reference_path = save_uploaded_audio(
                reference_file,
                temp_paths,
            )
            participant_path = save_uploaded_audio(
                participant_file,
                temp_paths,
            )

            with st.spinner(
                "Extracting acoustic features and aligning recordings..."
            ):
                reference_features = extract_features(reference_path)
                participant_features = extract_features(participant_path)

                comparison = align_features(
                    reference_features["mfcc"],
                    participant_features["mfcc"],
                )

                flaws = detect_flaws(
                    reference_features,
                    participant_features,
                )

            comparison_signature = (
                f"comparison:{get_upload_signature(reference_file)}:"
                f"{get_upload_signature(participant_file)}"
            )

            comparison_metrics = {
                "duration": float(participant_features["duration"]),
                "speech_seconds": None,
                "pause_count": None,
                "dtw_distance": float(comparison["normalized_cost"]),
                "flaw_count": len(flaws or []),
            }

            comparison_details = {
                "reference_duration": float(
                    reference_features["duration"]
                ),
                "participant_duration": float(
                    participant_features["duration"]
                ),
                "normalized_dtw_distance": float(
                    comparison["normalized_cost"]
                ),
                "flaws": flaws or [],
            }

            save_session(
                session_type="comparison",
                title=(
                    f"Comparison: {reference_file.name} "
                    f"vs {participant_file.name}"
                ),
                metrics=comparison_metrics,
                details=comparison_details,
                signature=comparison_signature,
            )

            st.success("Acoustic comparison complete!")

            st.divider()
            st.subheader("📊 Comparison Overview")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Reference Duration",
                f"{reference_features['duration']:.2f} sec",
            )

            col2.metric(
                "Participant Duration",
                f"{participant_features['duration']:.2f} sec",
            )

            col3.metric(
                "Normalized DTW Distance",
                f"{comparison['normalized_cost']:.3f}",
            )

            st.caption(
                "DTW distance measures acoustic dissimilarity along an "
                "alignment path; it is not a presentation-quality score."
            )

            st.divider()

            show_flaw_results(
                flaws,
                recording_duration=participant_features["duration"],
            )

            st.divider()

            show_speech_coach(flaws)

            st.divider()

            show_word_level_flaw_mapping(
                participant_path,
                flaws,
            )

            st.divider()

            show_diagnostics(
                reference_features,
                participant_features,
            )

            st.divider()
            st.subheader("🔗 Aligned Acoustic Features")

            path = comparison["path"]

            ref_times = librosa.frames_to_time(
                np.arange(reference_features["mfcc"].shape[1]),
                sr=16000,
                hop_length=160,
            )

            part_times = librosa.frames_to_time(
                np.arange(participant_features["mfcc"].shape[1]),
                sr=16000,
                hop_length=160,
            )

            aligned_ref = [
                float(ref_times[i])
                for i, j in path
            ]

            aligned_part = [
                float(part_times[j])
                for i, j in path
            ]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=aligned_ref,
                    y=aligned_part,
                    mode="lines",
                    name="DTW alignment",
                    line={"color": "#60a5fa", "width": 2},
                )
            )

            limit = min(
                reference_features["duration"],
                participant_features["duration"],
            )

            fig.add_trace(
                go.Scatter(
                    x=[0, limit],
                    y=[0, limit],
                    mode="lines",
                    name="Equal-time reference",
                    line={
                        "dash": "dash",
                        "color": "#a78bfa",
                    },
                )
            )

            fig.update_layout(
                title="Dynamic Time Warping Alignment Path",
                xaxis_title="Reference time (seconds)",
                yaxis_title="Participant time (seconds)",
                hovermode="closest",
            )

            style_chart(fig, 420)

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "The alignment path shows which acoustic frames were matched. "
                "The dashed line represents equal elapsed time."
            )

        except Exception as error:
            st.error(f"Comparison failed: {error}")
            st.exception(error)

        finally:
            for path_to_remove in temp_paths:
                if os.path.exists(path_to_remove):
                    try:
                        os.remove(path_to_remove)
                    except OSError:
                        pass

    elif not reference_file or not participant_file:
        st.markdown(
            """
            <div class="sw-flaw-card" style="text-align:center;padding:25px;">
                <div class="sw-flaw-title">
                    Ready to compare your recordings?
                </div>
                <div class="sw-flaw-meta">
                    Add both audio files above to start contrastive analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    show_footer()


# =========================================================
# PAGE 3 — PROGRESS TRACKING
# =========================================================

if page == "📈 Progress Tracking":

    show_hero(
        "Progress Tracking",
        "Review your previous speech analyses, compare acoustic trends "
        "and monitor your practice history over time.",
        "PERFORMANCE HISTORY",
    )

    sessions = get_sessions(limit=500)

    if not sessions:

        st.markdown(
            """
            <div class="sw-flaw-card" style="text-align:center;padding:35px;">
                <div style="font-size:35px;">📈</div>
                <div class="sw-flaw-title" style="font-size:17px;">
                    Your progress journey starts here
                </div>
                <div class="sw-flaw-meta">
                    Complete an individual analysis or recording comparison.
                    Your saved sessions will appear on this page.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        history_df = pd.DataFrame(sessions)

        history_df["created_at"] = pd.to_datetime(
            history_df["created_at"],
            errors="coerce",
        )

        history_df = history_df.sort_values("created_at")

        total_sessions = len(history_df)

        individual_count = int(
            (history_df["session_type"] == "individual").sum()
        )

        comparison_count = int(
            (history_df["session_type"] == "comparison").sum()
        )

        total_flaws = (
            int(history_df["flaw_count"].fillna(0).sum())
            if "flaw_count" in history_df.columns
            else 0
        )

        st.subheader("📊 Activity Overview")

        overview_cols = st.columns(4)

        overview_cols[0].metric(
            "Total Sessions",
            total_sessions,
        )

        overview_cols[1].metric(
            "Individual Analyses",
            individual_count,
        )

        overview_cols[2].metric(
            "Comparisons",
            comparison_count,
        )

        overview_cols[3].metric(
            "Potential Indicators",
            total_flaws,
        )

        st.divider()

        st.subheader("📈 Performance Trends")

        duration_df = history_df.dropna(subset=["duration"])

        if not duration_df.empty:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=duration_df["created_at"],
                    y=duration_df["duration"],
                    mode="lines+markers",
                    name="Duration",
                    text=duration_df["title"],
                    line={"color": "#60a5fa", "width": 2},
                    marker={"size": 8},
                )
            )

            fig.update_layout(
                title="Speech Duration Over Time",
                xaxis_title="Session date",
                yaxis_title="Duration (seconds)",
            )

            style_chart(fig)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Duration data is not available.")

        pause_df = (
            history_df.dropna(subset=["pause_count"])
            if "pause_count" in history_df.columns
            else pd.DataFrame()
        )

        if not pause_df.empty:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=pause_df["created_at"],
                    y=pause_df["pause_count"],
                    mode="lines+markers",
                    name="Pause count",
                    text=pause_df["title"],
                    line={"color": "#a78bfa", "width": 2},
                    marker={"size": 8},
                )
            )

            fig.update_layout(
                title="Pause Count Over Time",
                xaxis_title="Session date",
                yaxis_title="Pause count",
            )

            style_chart(fig)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Pause history is not available yet.")

        dtw_df = (
            history_df.dropna(subset=["dtw_distance"])
            if "dtw_distance" in history_df.columns
            else pd.DataFrame()
        )

        if not dtw_df.empty:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=dtw_df["created_at"],
                    y=dtw_df["dtw_distance"],
                    mode="lines+markers",
                    name="Normalized DTW distance",
                    text=dtw_df["title"],
                    line={"color": "#34d399", "width": 2},
                    marker={"size": 8},
                )
            )

            fig.update_layout(
                title="DTW Distance Over Time",
                xaxis_title="Session date",
                yaxis_title="Normalized DTW distance",
            )

            style_chart(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "With comparable recordings and consistent preprocessing, "
                "lower DTW distance indicates greater acoustic similarity. "
                "This is not a direct measure of speech quality."
            )

        else:
            st.info(
                "The DTW chart will appear after comparison sessions are saved."
            )

        st.divider()
        st.subheader("🗂️ Saved Session History")

        display_columns = [
            "id",
            "created_at",
            "session_type",
            "title",
            "duration",
            "speech_seconds",
            "pause_count",
            "dtw_distance",
            "flaw_count",
        ]

        available_columns = [
            col for col in display_columns
            if col in history_df.columns
        ]

        display_df = history_df[available_columns].copy()
        display_df["created_at"] = display_df["created_at"].astype(str)

        st.dataframe(
            display_df.sort_values("id", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.subheader("🔍 View or Delete a Session")

        session_ids = [
            int(session_id)
            for session_id in history_df["id"].dropna().tolist()
        ]

        selected_id = st.selectbox(
            "Select session ID",
            session_ids,
            format_func=lambda session_id: (
                f"Session #{session_id} — "
                + str(
                    next(
                        (
                            row["title"]
                            for row in sessions
                            if row["id"] == session_id
                        ),
                        "Saved session",
                    )
                )
            ),
            key="history_selected_session",
        )

        selected_session = get_session(selected_id)

        if selected_session:

            with st.expander(
                "📄 View session details",
                expanded=True,
            ):

                st.write(
                    "**Title:**",
                    selected_session.get("title"),
                )

                st.write(
                    "**Date:**",
                    selected_session.get("created_at"),
                )

                st.write(
                    "**Type:**",
                    selected_session.get("session_type"),
                )

                details = selected_session.get("details", {})

                st.json(details)

                saved_flaws = details.get("flaws", [])

                if saved_flaws:
                    st.markdown("#### 🔎 Saved Delivery Flaws")

                    st.dataframe(
                        pd.DataFrame(saved_flaws),
                        use_container_width=True,
                        hide_index=True,
                    )

        confirm_delete = st.checkbox(
            f"Confirm deletion of session #{selected_id}",
            key="confirm_delete_one",
        )

        if st.button(
            "🗑️ Delete Selected Session",
            disabled=not confirm_delete,
            key="delete_selected_session",
        ):
            delete_session(selected_id)
            st.success(f"Session #{selected_id} deleted.")
            st.rerun()

        st.divider()
        st.subheader("⚠️ Clear History")

        confirm_clear = st.checkbox(
            "I understand that this will delete all saved sessions.",
            key="confirm_clear_history",
        )

        if st.button(
            "Delete All History",
            disabled=not confirm_clear,
            key="clear_all_history",
        ):
            clear_history()
            st.success("All saved history has been deleted.")
            st.rerun()

    show_footer()
