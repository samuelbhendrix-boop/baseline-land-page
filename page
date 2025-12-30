# app.py
import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Baseline",
    page_icon="—",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------
# Minimal styling
# ----------------------------
st.markdown(
    """
<style>
/* Global typography + spacing */
html, body, [class*="css"]  {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
.block-container { padding-top: 2.5rem; padding-bottom: 4rem; max-width: 860px; }

/* Make buttons feel calmer */
.stButton>button {
  border-radius: 999px;
  padding: 0.7rem 1.0rem;
  border: 1px solid rgba(0,0,0,0.15);
}
.stButton>button:hover { border-color: rgba(0,0,0,0.35); }

/* Subtle section separators */
hr {
  border: none;
  border-top: 1px solid rgba(0,0,0,0.08);
  margin: 2rem 0;
}
.small { color: rgba(0,0,0,0.65); font-size: 0.92rem; }
.muted { color: rgba(0,0,0,0.65); }
.card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 18px;
  padding: 1.25rem 1.25rem;
  background: rgba(255,255,255,0.6);
}
.kicker { letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.78rem; color: rgba(0,0,0,0.55); }
.h1 { font-size: 2.25rem; line-height: 1.1; margin: 0.25rem 0 0.75rem 0; }
.h2 { font-size: 1.3rem; margin: 0 0 0.5rem 0; }
.sample-label { font-size: 0.9rem; color: rgba(0,0,0,0.6); margin-bottom: 0.35rem; }
.highlight {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  background: rgba(0,0,0,0.05);
  font-size: 0.95rem;
}
.footer { margin-top: 2.5rem; padding-top: 1.25rem; border-top: 1px solid rgba(0,0,0,0.08); }
a { color: inherit; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Top nav (simple)
# ----------------------------
nav1, nav2, nav3, nav4, nav5 = st.columns([2.2, 1, 1, 1, 1.3], vertical_alignment="center")
with nav1:
    st.markdown("**Baseline**")
with nav2:
    st.markdown('<span class="small"><a href="#how-it-works">How it works</a></span>', unsafe_allow_html=True)
with nav3:
    st.markdown('<span class="small"><a href="#trust">Trust</a></span>', unsafe_allow_html=True)
with nav4:
    st.markdown('<span class="small"><a href="#pricing">Pricing</a></span>', unsafe_allow_html=True)
with nav5:
    cta_top = st.button("Get your Baseline", use_container_width=True)

# ----------------------------
# State
# ----------------------------
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False
if cta_top:
    st.session_state.show_signup = True

# ----------------------------
# Hero
# ----------------------------
st.markdown('<div class="kicker">A weekly financial instrument</div>', unsafe_allow_html=True)
st.markdown('<div class="h1">Your financial life, clarified.</div>', unsafe_allow_html=True)
st.markdown(
    "Once a week, Baseline tells you whether your financial trajectory is actually getting better or worse — and why.\n\n"
    "**No dashboards. No budgeting. No daily tracking.**"
)

hero_cta1, hero_cta2 = st.columns([1.2, 1], vertical_alignment="center")
with hero_cta1:
    if st.button("Get your Baseline", type="primary", use_container_width=True):
        st.session_state.show_signup = True
with hero_cta2:
    st.markdown('<span class="small">Works without linking your bank account.</span>', unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# What you get
# ----------------------------
st.markdown("## One number. One sentence. Once a week.")
colA, colB = st.columns(2)
with colA:
    st.markdown("**Months of Freedom** — how long you could maintain your current lifestyle if income stopped")
    st.markdown("**Direction** — meaningful week-over-week change (noise filtered)")
with colB:
    st.markdown("**Cause** — the one thing that actually moved the needle")
    st.markdown("**Leverage** — what matters if the pattern continues")

# ----------------------------
# Sample week
# ----------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("## Sample Week")
st.markdown(
    """
<div class="card">
  <div class="sample-label">Trajectory</div>
  You’re at <span class="highlight"><b>4.4 months of freedom</b></span>, up <span class="highlight"><b>+0.3</b></span> from last week.
  <br/><br/>
  <div class="sample-label">What changed</div>
  The improvement came mainly from <b>fewer late-evening convenience purchases</b> than usual.
  <br/><br/>
  <div class="sample-label">If this holds</div>
  Keeping evenings this way could add about <b>+1.1 months of freedom</b> over a year.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# How it works
# ----------------------------
st.markdown('<a id="how-it-works"></a>', unsafe_allow_html=True)
st.markdown("## How it works")
step1, step2, step3 = st.columns(3)
with step1:
    st.markdown("**1) Set your Baseline**")
    st.markdown("Enter two numbers (rough is fine):\n- monthly spending (estimate)\n- liquid cash (checking + savings)")
with step2:
    st.markdown("**2) Baseline observes quietly**")
    st.markdown("Optional: connect **one** checking account or upload a CSV for better accuracy.\n\nBaseline doesn’t need your entire financial life.")
with step3:
    st.markdown("**3) Receive your weekly Baseline**")
    st.markdown("A single readout designed to **reduce thinking**, not add to it.")

# ----------------------------
# Trust section
# ----------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<a id="trust"></a>', unsafe_allow_html=True)
st.markdown("## How Baseline earns trust")
st.markdown(
    """
- **Works without bank connections** (manual mode is always supported)  
- **Asks for the minimum data required**  
- **Easy to disconnect anytime**  
- **Subscription-funded** — no selling, sharing, or training on your financial data  
"""
)
st.markdown('<p class="small muted">Baseline avoids investments, net worth dashboards, and predictions on purpose.</p>', unsafe_allow_html=True)

# ----------------------------
# Who it's for / not for
# ----------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("## Who it’s for")
w1, w2 = st.columns(2)
with w1:
    st.markdown("### Baseline is for you if")
    st.markdown(
        "- you’re doing “fine” but still feel financial noise\n"
        "- you hate budgeting apps and spreadsheets\n"
        "- you want truth, not motivation\n"
        "- you’d rather check once a week and move on"
    )
with w2:
    st.markdown("### Baseline isn’t for you if")
    st.markdown(
        "- you want daily optimization and alerts\n"
        "- you want category breakdowns and dashboards\n"
        "- you’re looking for investing advice"
    )

# ----------------------------
# Pricing
# ----------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<a id="pricing"></a>', unsafe_allow_html=True)
st.markdown("## Pricing")

st.markdown(
    """
<div class="card">
  <div style="display:flex; align-items:baseline; justify-content:space-between; gap:1rem; flex-wrap:wrap;">
    <div>
      <div class="kicker">Subscription</div>
      <div style="font-size:2rem; line-height:1.1; margin-top:0.3rem;"><b>$24/month</b></div>
      <div class="small muted" style="margin-top:0.25rem;">First 3 weeks are free (calibration + your first signal). Cancel anytime.</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if st.button("Get your Baseline", type="primary", use_container_width=True):
    st.session_state.show_signup = True

# ----------------------------
# FAQ
# ----------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("## FAQ")

with st.expander("Do I need to link my bank account?"):
    st.write("No. Baseline works with manual inputs. You can optionally connect one account later for cleaner signals.")
with st.expander("How does Baseline measure “Months of Freedom”?"):
    st.write("It uses your liquid cash and your smoothed lifestyle burn (based on your inputs and/or transaction history).")
with st.expander("Is Baseline AI?"):
    st.write("Baseline uses conservative pattern detection to explain meaningful changes. It prioritizes stability over cleverness.")
with st.expander("Will this tell me what to do?"):
    st.write("Baseline doesn’t give financial advice. It gives a clear readout of trajectory and what changed.")
with st.expander("What happens if a week is unusual (travel, emergencies)?"):
    st.write("Baseline flags non-representative weeks and avoids drawing conclusions from noise.")

# ----------------------------
# Signup modal-ish section
# ----------------------------
if st.session_state.show_signup:
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("## Get your Baseline")

    with st.form("signup_form", clear_on_submit=False):
        st.markdown('<p class="small muted">Start in manual mode. You can connect an account later (optional).</p>', unsafe_allow_html=True)

        email = st.text_input("Email", placeholder="you@domain.com")

        c1, c2 = st.columns(2)
        with c1:
            monthly_spend = st.number_input("Monthly spending (estimate)", min_value=0, value=4200, step=50)
        with c2:
            liquid_cash = st.number_input("Liquid cash (checking + savings)", min_value=0, value=18000, step=100)

        st.caption("Rough is fine. Baseline becomes clearer over a few weeks.")
        submitted = st.form_submit_button("Start (Free for 3 weeks)", type="primary")

    if submitted:
        if not email.strip():
            st.error("Please enter an email.")
        else:
            # Demo-only: In production, store this in a database and kick off onboarding email.
            st.success("You're in. Baseline will send your first calibration email shortly (demo behavior).")
            st.markdown(
                f"""
<div class="card">
  <div class="sample-label">Saved (demo)</div>
  <b>Email:</b> {email}<br/>
  <b>Monthly spend:</b> ${monthly_spend:,.0f}<br/>
  <b>Liquid cash:</b> ${liquid_cash:,.0f}<br/>
</div>
""",
                unsafe_allow_html=True,
            )

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
<div class="footer small muted">
  <div style="display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap;">
    <div>© Baseline</div>
    <div>
      <a href="#trust">Privacy</a> · <a href="#pricing">Terms</a> · <a href="#how-it-works">Contact</a>
    </div>
  </div>
  <div style="margin-top:0.5rem;">Built with restraint.</div>
</div>
""",
    unsafe_allow_html=True,
)
