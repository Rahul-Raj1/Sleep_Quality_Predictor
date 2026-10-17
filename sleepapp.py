import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------ page setup
st.set_page_config(page_title="Sleep Quality Predictor", page_icon="🌙", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.5rem; max-width: 1150px;}
#MainMenu, footer {visibility: hidden;}

.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 55%, #7c3aed 100%);
    border-radius: 22px; padding: 38px 42px; color: #fff; margin-bottom: 24px;
    box-shadow: 0 12px 30px rgba(67, 56, 202, .35);
}
.hero h1 {margin: 0; font-size: 2.5rem; color: #fff;}
.hero p  {margin: 8px 0 0; font-size: 1.1rem; color: #e0e7ff;}

.card {
    background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.25);
    border-radius: 18px; padding: 22px 24px; margin-bottom: 16px;
}
.card h3 {margin-top: 0;}

.badge {
    display: inline-block; padding: 8px 20px; border-radius: 999px;
    font-weight: 700; font-size: 1.15rem; color: #fff;
}
.tile {
    border-radius: 16px; padding: 16px; text-align: center;
    background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.25);
}
.tile .icon {font-size: 1.8rem;}
.tile .val  {font-size: 1.5rem; font-weight: 700;}
.tile .lbl  {opacity: .7; font-size: .85rem;}
.chip {
    display: inline-block; margin-top: 6px; padding: 2px 12px; border-radius: 999px;
    font-size: .78rem; font-weight: 600; color: #fff;
}
.tip {
    border-radius: 14px; padding: 14px 18px; margin-bottom: 12px;
    background: rgba(128,128,128,.07); border-left: 6px solid #888;
}
.tip b {font-size: 1.02rem;}
.tip.good {border-left-color: #22c55e;}
.tip.warn {border-left-color: #f59e0b;}
.tip.bad  {border-left-color: #ef4444;}
.small {opacity: .7; font-size: .88rem;}
.result {text-align: center; padding-bottom: 20px;}
.pills {margin-top: 6px;}
.pill {
    display: inline-block; margin: 4px 6px; padding: 5px 16px; border-radius: 999px;
    border: 2px solid; font-weight: 700; font-size: .95rem; background: rgba(128,128,128,.06);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

GREEN, AMBER, RED = "#22c55e", "#f59e0b", "#ef4444"


# ------------------------------------------------------------------ model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl"), joblib.load("feature_columns.pkl")


try:
    model, feature_cols = load_model()
except FileNotFoundError:
    st.error("model.pkl not found. Run Sleep_Quality_Prediction.ipynb first so that "
             "model.pkl and feature_columns.pkl are created in this folder.")
    st.stop()

LABELS = {"age": "Age", "screen_time": "Screen time",
          "caffeine_intake": "Caffeine", "exercise_hours": "Exercise"}


def build_row(age, gender, screen, caffeine, exercise):
    row = pd.DataFrame([{
        "age": age, "screen_time": screen, "caffeine_intake": caffeine,
        "exercise_hours": exercise,
        "gender_Male": int(gender == "Male"), "gender_Other": int(gender == "Other"),
    }])
    return row[feature_cols]


def predict_prob(age, gender, screen, caffeine, exercise):
    return float(model.predict_proba(build_row(age, gender, screen, caffeine, exercise))[0, 1])


def factor_impacts(row):
    scaled = model.named_steps["scaler"].transform(row)[0]
    coef = model.named_steps["clf"].coef_[0]
    s = pd.Series(coef * scaled, index=feature_cols)
    s = s[list(LABELS)]
    s.index = [LABELS[i] for i in s.index]
    return s.sort_values()


# ------------------------------------------------------------------ UI helpers
def score_color(p):
    return GREEN if p >= 0.65 else AMBER if p >= 0.5 else RED


def H(s):
    """Collapse HTML to one line. Markdown treats indented lines and blank lines as code blocks."""
    return "".join(line.strip() for line in s.splitlines())


def gauge_svg(p):
    import math
    cx, cy, r = 100, 100, 85
    length = math.pi * r
    ang = math.pi * (1 - p)
    ex, ey = cx + r * math.cos(ang), cy - r * math.sin(ang)
    color = score_color(p)
    return H(f"""
    <div style="text-align:center">
    <svg viewBox="0 0 200 128" width="100%" style="max-width:360px">
      <defs>
        <linearGradient id="gaugeGrad" gradientUnits="userSpaceOnUse" x1="15" y1="0" x2="185" y2="0">
          <stop offset="0%" stop-color="#ef4444"/>
          <stop offset="50%" stop-color="#f59e0b"/>
          <stop offset="100%" stop-color="#22c55e"/>
        </linearGradient>
        <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="2.5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <path d="M15 100 A85 85 0 0 1 185 100" fill="none" stroke="rgba(128,128,128,.22)"
            stroke-width="14" stroke-linecap="round"/>
      <path d="M15 100 A85 85 0 0 1 185 100" fill="none" stroke="url(#gaugeGrad)"
            stroke-width="14" stroke-linecap="round" filter="url(#glow)"
            stroke-dasharray="{p * length:.1f} {length:.1f}"/>
      <circle cx="{ex:.1f}" cy="{ey:.1f}" r="8" fill="#fff" stroke="{color}" stroke-width="4"/>
      <text x="100" y="92" text-anchor="middle" font-size="40" font-weight="800" fill="{color}">{p * 100:.0f}</text>
      <text x="100" y="110" text-anchor="middle" font-size="9" letter-spacing="2" fill="gray">SLEEP SCORE</text>
      <text x="15" y="122" text-anchor="middle" font-size="8" fill="gray">0</text>
      <text x="185" y="122" text-anchor="middle" font-size="8" fill="gray">100</text>
    </svg>
    </div>""")


def tile(icon, label, value, status, color):
    return H(f"""
    <div class="tile">
      <div class="icon">{icon}</div>
      <div class="val">{value}</div>
      <div class="lbl">{label}</div>
      <span class="chip" style="background:{color}">{status}</span>
    </div>""")


def status_screen(h):
    return ("Healthy", GREEN) if h <= 4 else ("Moderate", AMBER) if h <= 6 else ("High", RED)


def status_caffeine(c):
    return ("Healthy", GREEN) if c <= 2 else ("Moderate", AMBER) if c <= 3 else ("High", RED)


def status_exercise(h):
    return ("Great", GREEN) if h >= 1 else ("Low", AMBER) if h >= 0.5 else ("Very low", RED)


def get_advice(p, screen, caffeine, exercise, age):
    tips = []
    if screen > 6:
        tips.append(("bad", "📱 Cut your screen time",
                     f"{screen:.1f} h/day is high. Aim for under 4-5 h, switch off screens 60 minutes "
                     "before bed and use night mode in the evening."))
    elif screen > 4:
        tips.append(("warn", "📱 Trim screen time a little",
                     "Moderate use. Set a screen curfew an hour before bedtime."))
    else:
        tips.append(("good", "📱 Screen time looks healthy", "Keep screens away from the bedroom."))

    if caffeine >= 4:
        tips.append(("bad", "☕ Reduce caffeine",
                     f"{caffeine} cups/day is a lot. Step down to 2 cups or fewer and take your last "
                     "cup before 2 PM. Caffeine can stay active for 6+ hours."))
    elif caffeine == 3:
        tips.append(("warn", "☕ Watch caffeine timing", "Keep it to mornings and early afternoon."))
    else:
        tips.append(("good", "☕ Caffeine is under control", "Keep your intake at 2 cups or fewer."))

    if exercise < 0.5:
        tips.append(("bad", "🏃 Start moving",
                     "Begin with a 20-30 minute daily walk and build up to about 1 hour a day. "
                     "Avoid intense workouts right before bed."))
    elif exercise < 1:
        tips.append(("warn", "🏃 Add a little more exercise", "You are close. Aim for about 1 hour a day."))
    else:
        tips.append(("good", "🏃 Great exercise habit", "Regular activity deepens sleep. Keep it up."))

    if p < 0.5:
        tips.append(("warn", "🌙 Build a sleep routine",
                     "Sleep and wake at the same time daily, keep the room dark, cool and quiet, "
                     "and wind down with stretching, reading or deep breathing."))
    else:
        tips.append(("good", "🌙 Protect your routine",
                     "Keep a consistent sleep and wake time, even on weekends."))
    if age >= 45:
        tips.append(("warn", "🧓 Age-friendly tip",
                     "Sleep gets lighter with age. Limit late fluids and heavy dinners, and get morning sunlight."))
    return tips


# ------------------------------------------------------------------ header
st.markdown("""
<div class="hero">
  <h1>😴 Sleep Quality Predictor </h1>
  <p>Predict your sleep quality from your daily habits and get personalised advice to sleep better.</p>
</div>""", unsafe_allow_html=True)

tab_predict, tab_whatif, tab_tips, tab_about = st.tabs(
    ["🔮 Predict", "🧪 What-if Simulator", "💡 Sleep Tips", "ℹ️ About"])

# ------------------------------------------------------------------ TAB 1: predict
with tab_predict:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown('<div class="card"><h3>📝 Your daily habits</h3>', unsafe_allow_html=True)
        with st.form("habits"):
            c1, c2 = st.columns(2)
            age = c1.slider("🎂 Age", 18, 60, 30)
            gender = c2.selectbox("🧑 Gender", ["Female", "Male", "Other"])
            screen = st.slider("📱 Screen time (hours/day)", 0.0, 12.0, 5.0, 0.5)
            caffeine = st.slider("☕ Caffeine (cups/day)", 0, 8, 2)
            exercise = st.slider("🏃 Exercise (hours/day)", 0.0, 3.0, 1.0, 0.25)
            submitted = st.form_submit_button("✨ Predict my sleep quality",
                                              type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        st.session_state["inputs"] = dict(age=age, gender=gender, screen=screen,
                                          caffeine=caffeine, exercise=exercise)

    with right:
        if "inputs" not in st.session_state:
            st.markdown('<div class="card"><h3>👈 Get started</h3>'
                        '<p>Set your habits on the left and press <b>Predict my sleep quality</b>. '
                        'Your score, key factors and advice will appear here.</p></div>',
                        unsafe_allow_html=True)
        else:
            d = st.session_state["inputs"]
            p = predict_prob(d["age"], d["gender"], d["screen"], d["caffeine"], d["exercise"])
            good = p >= 0.5
            color = score_color(p)

            msg = ("Great news! Your habits support restful sleep." if p >= 0.65 else
                   "You are on the edge. Small changes can tip the balance." if p >= 0.5 else
                   "Your current habits are working against your sleep. See the advice below.")
            st.markdown(H(f"""
            <div class="card result">
              <span class="badge" style="background:{color}">
                {'😊 Good sleep likely' if good else '😟 Poor sleep likely'}</span>
              {gauge_svg(p)}
              <div class="pills">
                <span class="pill" style="border-color:{GREEN};color:{GREEN}">👍 Good sleep &nbsp;{p:.0%}</span>
                <span class="pill" style="border-color:{RED};color:{RED}">👎 Poor sleep &nbsp;{1 - p:.0%}</span>
              </div>
              <div class="small" style="margin-top:12px">{msg}</div>
            </div>"""), unsafe_allow_html=True)

            t1, t2, t3 = st.columns(3)
            s_txt, s_col = status_screen(d["screen"])
            c_txt, c_col = status_caffeine(d["caffeine"])
            e_txt, e_col = status_exercise(d["exercise"])
            t1.markdown(tile("📱", "Screen time", f"{d['screen']:.1f} h", s_txt, s_col), unsafe_allow_html=True)
            t2.markdown(tile("☕", "Caffeine", f"{d['caffeine']} cups", c_txt, c_col), unsafe_allow_html=True)
            t3.markdown(tile("🏃", "Exercise", f"{d['exercise']:.2f} h", e_txt, e_col), unsafe_allow_html=True)

    if "inputs" in st.session_state:
        d = st.session_state["inputs"]
        p = predict_prob(d["age"], d["gender"], d["screen"], d["caffeine"], d["exercise"])

        st.markdown("---")
        a_col, b_col = st.columns([1, 1.15], gap="large")

        with a_col:
            st.subheader("📊 What is affecting your result?")
            imp = factor_impacts(build_row(d["age"], d["gender"], d["screen"], d["caffeine"], d["exercise"]))
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(imp.index, imp.values, color=[GREEN if v >= 0 else RED for v in imp.values])
            ax.axvline(0, color="gray", lw=0.8)
            ax.set_xlabel("← pushes to Poor        pushes to Good →")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            ax.tick_params(colors="gray")
            ax.xaxis.label.set_color("gray")
            st.pyplot(fig, use_container_width=True)

        with b_col:
            st.subheader("💡 Your personalised advice")
            for kind, title, text in get_advice(p, d["screen"], d["caffeine"], d["exercise"], d["age"]):
                st.markdown(f'<div class="tip {kind}"><b>{title}</b><br>{text}</div>',
                            unsafe_allow_html=True)

# ------------------------------------------------------------------ TAB 2: what-if
with tab_whatif:
    st.markdown("### 🧪 See how better habits change your score")
    if "inputs" not in st.session_state:
        st.info("First make a prediction in the **Predict** tab, then come back here.")
    else:
        d = st.session_state["inputs"]
        base = predict_prob(d["age"], d["gender"], d["screen"], d["caffeine"], d["exercise"])
        w1, w2, w3 = st.columns(3)
        n_screen = w1.slider("New screen time (h)", 0.0, 12.0, float(d["screen"]), 0.5, key="ns")
        n_caff = w2.slider("New caffeine (cups)", 0, 8, int(d["caffeine"]), key="nc")
        n_ex = w3.slider("New exercise (h)", 0.0, 3.0, float(d["exercise"]), 0.25, key="ne")
        new = predict_prob(d["age"], d["gender"], n_screen, n_caff, n_ex)

        m1, m2, m3 = st.columns(3)
        m1.metric("Current score", f"{base * 100:.0f} / 100")
        m2.metric("New score", f"{new * 100:.0f} / 100", f"{(new - base) * 100:+.0f} points")
        m3.metric("New prediction", "Good 😊" if new >= 0.5 else "Poor 😟")
        st.progress(new, text=f"New sleep score: {new * 100:.0f} / 100")

# ------------------------------------------------------------------ TAB 3: tips
with tab_tips:
    st.markdown("### 💡 Daily sleep habits that help")
    c1, c2 = st.columns(2)
    c1.markdown("""<div class="card"><h3>✅ Do</h3>
    <ul><li>Keep a fixed sleep and wake time</li>
    <li>Get morning sunlight and daily exercise</li>
    <li>Keep the bedroom dark, cool and quiet</li>
    <li>Wind down 30-60 minutes before bed</li>
    <li>Stop caffeine after 2 PM</li></ul></div>""", unsafe_allow_html=True)
    c2.markdown("""<div class="card"><h3>🚫 Avoid</h3>
    <ul><li>Phones and laptops in bed</li>
    <li>Heavy meals or alcohol late at night</li>
    <li>Intense workouts right before sleep</li>
    <li>Long afternoon naps</li>
    <li>Checking the clock repeatedly at night</li></ul></div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ TAB 4: about
with tab_about:
    st.markdown("""
    ### ℹ️ About this project
    **Model:** Logistic Regression (scikit-learn) with standardised features.
    **Inputs:** age, gender, screen time, caffeine intake and exercise hours.
    **Output:** Good or Poor sleep quality, with the probability shown as a sleep score.
    **Data:** a synthetic dataset of about 420 records, created for this learning project.

    > ⚠️ This app is for education only and is **not medical advice**.
    > If you have long-term sleep problems, please speak to a doctor.
    """)

st.markdown('<p class="small" style="text-align:center;margin-top:30px">'
            'Made with ❤️ using Streamlit & scikit-learn</p>', unsafe_allow_html=True)