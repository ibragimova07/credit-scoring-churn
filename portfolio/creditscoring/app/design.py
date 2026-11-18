"""Powder Blue Minimalism — design system for the churn Streamlit app.

All visual tokens live here (and in ``.streamlit/config.toml`` for the native
Streamlit theme).  This module injects one global stylesheet and exposes a few
small presentational helpers so every section of the app shares the same
look and feel.  No business logic lives in this file.
"""
from __future__ import annotations

from contextlib import contextmanager
from html import escape

import streamlit as st

# --------------------------------------------------------------------------- #
# Design tokens (kept in sync with .streamlit/config.toml)
# --------------------------------------------------------------------------- #
TOKENS = {
    "primary": "#A9C7E8",
    "primary_hover": "#8FB4DC",
    "primary_light": "#EAF2FA",
    "background": "#F7F9FC",
    "surface": "#FFFFFF",
    "surface_secondary": "#F0F4F8",
    "text_primary": "#26364A",
    "text_secondary": "#718096",
    "border": "#E3EAF1",
    "success": "#6FAF91",
    "warning": "#E7B66B",
    "error": "#D98282",
}

# --------------------------------------------------------------------------- #
# Global stylesheet
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---- tokens -------------------------------------------------------------- */
:root {
  --pb-primary:        #A9C7E8;
  --pb-primary-hover:  #8FB4DC;
  --pb-primary-light:  #EAF2FA;
  --pb-bg:             #F7F9FC;
  --pb-surface:        #FFFFFF;
  --pb-surface-2:      #F0F4F8;
  --pb-text:           #26364A;
  --pb-text-2:         #718096;
  /* Accessible darker shade of Text Secondary for small body text (WCAG AA). */
  --pb-text-2-strong:  #5F6E82;
  --pb-border:         #E3EAF1;
  --pb-success:        #6FAF91;
  --pb-warning:        #E7B66B;
  --pb-error:          #D98282;

  --pb-r-sm:  8px;
  --pb-r-md:  12px;
  --pb-r-lg:  16px;
  --pb-pill:  999px;

  --pb-sp-1:   4px;
  --pb-sp-2:   8px;
  --pb-sp-3:  12px;
  --pb-sp-4:  16px;
  --pb-sp-5:  20px;
  --pb-sp-6:  24px;
  --pb-sp-8:  32px;
  --pb-sp-12: 48px;
  --pb-sp-16: 64px;

  --pb-shadow-sm: 0 1px 2px rgba(38, 54, 74, 0.05);
  --pb-shadow:    0 1px 2px rgba(38, 54, 74, 0.04), 0 8px 24px rgba(38, 54, 74, 0.05);
  --pb-focus:     0 0 0 3px rgba(169, 199, 232, 0.45);
}

/* ---- foundation ---------------------------------------------------------- */
html, body, [data-testid="stApp"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
               'Helvetica Neue', Arial, sans-serif !important;
  color: var(--pb-text);
}
[data-testid="stAppViewContainer"] {
  background: var(--pb-bg);
}
.main .block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
}
code, kbd, .stCode {
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace !important;
  font-size: 0.8em;
  color: #3E6288;
  background: var(--pb-primary-light);
  padding: 1px 5px;
  border-radius: 5px;
}
hr { border-color: var(--pb-border) !important; }

/* ---- type scale ---------------------------------------------------------- */
h1, h2, h3, h4 {
  color: var(--pb-text);
  font-weight: 600 !important;
  letter-spacing: -0.02em;
}
h1 { font-size: 1.55rem !important; }
h2 { font-size: 1.12rem !important; margin-top: 0.25rem !important; }
h3 { font-size: 0.92rem !important; }
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p {
  font-size: 0.88rem !important;
  line-height: 1.6;
}
[data-testid="stCaptionContainer"] {
  color: var(--pb-text-2-strong) !important;
  font-size: 0.76rem !important;
}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] > div {
  color: var(--pb-text-2-strong) !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.02em;
}

/* ---- top bar ------------------------------------------------------------- */
[data-testid="stHeader"] {
  background: rgba(247, 249, 252, 0.88) !important;
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--pb-border);
}

/* ---- sidebar ------------------------------------------------------------- */
section[data-testid="stSidebar"] {
  background: var(--pb-surface) !important;
  border-right: 1px solid var(--pb-border) !important;
  box-shadow: 1px 0 0 rgba(38, 54, 74, 0.015);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; }

.pb-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 4px 4px 20px 4px;
  margin-bottom: 4px;
  border-bottom: 1px solid var(--pb-border);
}
.pb-brand-mark {
  display: grid;
  place-items: center;
  width: 36px;
  height: 36px;
  flex: 0 0 36px;
  border-radius: 10px;
  background: linear-gradient(145deg, var(--pb-primary) 0%, #93B7DE 100%);
  color: var(--pb-text);
  font-weight: 700;
  font-size: 0.85rem;
  box-shadow: var(--pb-shadow-sm);
}
.pb-brand-name {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--pb-text);
  line-height: 1.2;
}
.pb-brand-sub {
  font-size: 0.66rem;
  color: var(--pb-text-2-strong);
  margin-top: 2px;
}

/* Sidebar radio = primary navigation --------------------------------------- */
section[data-testid="stSidebar"] [role="radiogroup"] {
  display: flex;
  flex-direction: column;
  gap: var(--pb-sp-1);
  padding-top: var(--pb-sp-2);
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px !important;
  border-radius: var(--pb-r-sm);
  border-left: 3px solid transparent;
  color: var(--pb-text-2-strong);
  font-size: 0.8rem;
  cursor: pointer;
  transition: background 180ms ease, color 180ms ease, border-color 180ms ease;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: var(--pb-primary-light);
  color: var(--pb-text);
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[aria-checked="true"]),
section[data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
  background: var(--pb-primary-light);
  color: var(--pb-text);
  border-left-color: var(--pb-primary-hover);
  font-weight: 600;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
  accent-color: var(--pb-primary-hover);
  width: 15px;
  height: 15px;
  margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:focus-within {
  outline: 3px solid rgba(169, 199, 232, 0.55);
  outline-offset: 1px;
}

/* ---- buttons ------------------------------------------------------------- */
[data-testid="stButton"] button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {
  border-radius: var(--pb-r-sm) !important;
  border: 1px solid var(--pb-primary-hover) !important;
  background: var(--pb-primary) !important;
  color: var(--pb-text) !important;
  font-weight: 600 !important;
  font-size: 0.8rem !important;
  min-height: 38px;
  padding: 8px 18px !important;
  box-shadow: var(--pb-shadow-sm) !important;
  transition: background 180ms ease, border-color 180ms ease,
              box-shadow 180ms ease, transform 120ms ease !important;
}
[data-testid="stButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
  background: var(--pb-primary-hover) !important;
  border-color: #7FA6D2 !important;
  box-shadow: 0 4px 12px rgba(38, 54, 74, 0.10) !important;
}
[data-testid="stButton"] button:active,
[data-testid="stFormSubmitButton"] button:active,
[data-testid="stDownloadButton"] button:active {
  transform: translateY(1px);
  box-shadow: var(--pb-shadow-sm) !important;
}
[data-testid="stButton"] button:focus-visible,
[data-testid="stFormSubmitButton"] button:focus-visible,
[data-testid="stDownloadButton"] button:focus-visible {
  outline: none !important;
  box-shadow: var(--pb-focus) !important;
}
button:disabled { opacity: 0.5 !important; cursor: not-allowed !important; }

/* Secondary action: download ---------------------------------------------- */
[data-testid="stDownloadButton"] button {
  background: var(--pb-surface) !important;
  border-color: var(--pb-border) !important;
  color: #3E6288 !important;
}
[data-testid="stDownloadButton"] button:hover {
  background: var(--pb-primary-light) !important;
  border-color: var(--pb-primary) !important;
}

/* ---- forms & inputs ------------------------------------------------------ */
[data-testid="stForm"] {
  background: var(--pb-surface) !important;
  border: 1px solid var(--pb-border) !important;
  border-radius: var(--pb-r-lg) !important;
  padding: var(--pb-sp-6) !important;
  box-shadow: var(--pb-shadow) !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input {
  background: var(--pb-surface) !important;
  border: 1px solid var(--pb-border) !important;
  border-radius: var(--pb-r-sm) !important;
  color: var(--pb-text) !important;
  font-size: 0.84rem !important;
  min-height: 38px;
  padding: 7px 11px !important;
  box-shadow: none !important;
  transition: border-color 150ms ease, box-shadow 150ms ease;
}
[data-testid="stNumberInput"] input:hover,
[data-testid="stTextInput"] input:hover,
[data-testid="stTextArea"] textarea:hover {
  border-color: var(--pb-primary) !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--pb-primary-hover) !important;
  box-shadow: var(--pb-focus) !important;
  outline: none !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: var(--pb-surface) !important;
  border: 1px solid var(--pb-border) !important;
  border-radius: var(--pb-r-sm) !important;
  min-height: 38px;
  box-shadow: none !important;
  color: var(--pb-text) !important;
  font-size: 0.84rem !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
  border-color: var(--pb-primary-hover) !important;
  box-shadow: var(--pb-focus) !important;
}
[data-baseweb="popover"] { border-radius: var(--pb-r-sm) !important; }
[role="option"]:hover, [role="option"][aria-selected="true"] {
  background: var(--pb-primary-light) !important;
  color: var(--pb-text) !important;
}
input[type="checkbox"] { accent-color: var(--pb-primary-hover) !important; }

/* ---- file uploader ------------------------------------------------------- */
[data-testid="stFileUploader"] section {
  background: var(--pb-primary-light) !important;
  border: 1.5px dashed var(--pb-primary) !important;
  border-radius: var(--pb-r-md) !important;
  padding: var(--pb-sp-6) !important;
  transition: background 180ms ease, border-color 180ms ease;
}
[data-testid="stFileUploader"] section:hover {
  background: #E4EFFA !important;
  border-color: var(--pb-primary-hover) !important;
}

/* ---- metrics ------------------------------------------------------------- */
[data-testid="stMetric"] {
  background: var(--pb-surface) !important;
  border: 1px solid var(--pb-border) !important;
  border-radius: var(--pb-r-md) !important;
  padding: var(--pb-sp-4) var(--pb-sp-6) !important;
  box-shadow: var(--pb-shadow) !important;
  transition: transform 180ms ease, box-shadow 180ms ease;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(38, 54, 74, 0.08) !important;
}
[data-testid="stMetricLabel"] {
  color: var(--pb-text-2-strong) !important;
  font-size: 0.7rem !important;
  font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
  color: var(--pb-text) !important;
  font-size: 1.3rem !important;
  font-weight: 600 !important;
  letter-spacing: -0.02em;
}
[data-testid="stMetricDelta"] { font-size: 0.7rem !important; }

/* ---- figures ------------------------------------------------------------- */
/* Figures are intentionally smaller than full width so charts stay legible
   and don't dominate the page. They still shrink fluidly on small screens. */
[data-testid="stImage"] {
  max-width: 760px;
  margin-left: auto;
  margin-right: auto;
}
[data-testid="stImage"] img {
  display: block;
  width: 100% !important;
  max-width: 760px !important;
  height: auto !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

/* ---- tables -------------------------------------------------------------- */
[data-testid="stDataFrame"] {
  border: 1px solid var(--pb-border) !important;
  border-radius: var(--pb-r-md) !important;
  overflow: hidden;
  box-shadow: var(--pb-shadow-sm) !important;
}

/* ---- progress ------------------------------------------------------------ */
[data-testid="stProgress"] [role="progressbar"] {
  background: var(--pb-primary-light) !important;
  border-radius: var(--pb-pill) !important;
  height: 8px !important;
}
[data-testid="stProgress"] [role="progressbar"] > div {
  background: linear-gradient(90deg, var(--pb-primary), var(--pb-primary-hover)) !important;
  border-radius: var(--pb-pill) !important;
}

/* ---- alerts & status messages -------------------------------------------- */
[data-testid="stAlert"] {
  border-radius: var(--pb-r-sm) !important;
  border-left-width: 4px !important;
  border-left-style: solid !important;
  padding: var(--pb-sp-3) var(--pb-sp-4) !important;
  font-size: 0.8rem !important;
}
[data-testid="stAlert"][data-type="success"],
div[data-testid="stAlert"]:has(svg[data-testid="stIconSuccess"]) {
  border-left-color: var(--pb-success) !important;
  background: #EEF7F2 !important;
  color: #366B53 !important;
}
[data-testid="stAlert"][data-type="error"],
div[data-testid="stAlert"]:has(svg[data-testid="stIconError"]) {
  border-left-color: var(--pb-error) !important;
  background: #FCF0F0 !important;
  color: #934545 !important;
}
[data-testid="stAlert"][data-type="warning"],
div[data-testid="stAlert"]:has(svg[data-testid="stIconWarning"]) {
  border-left-color: var(--pb-warning) !important;
  background: #FDF6E9 !important;
  color: #8A5E1A !important;
}
[data-testid="stAlert"] p { color: inherit !important; }

/* ---- presentational helpers ---------------------------------------------- */
.pb-hero {
  position: relative;
  background: linear-gradient(135deg, var(--pb-surface) 0%, var(--pb-primary-light) 100%);
  border: 1px solid var(--pb-border);
  border-radius: var(--pb-r-lg);
  padding: var(--pb-sp-8) var(--pb-sp-8);
  box-shadow: var(--pb-shadow);
  margin-bottom: var(--pb-sp-8);
  overflow: hidden;
}
.pb-hero::after {
  content: "";
  position: absolute;
  right: -40px;
  top: -60px;
  width: 220px;
  height: 220px;
  border-radius: 50%;
  background: rgba(169, 199, 232, 0.18);
  pointer-events: none;
}
.pb-eyebrow {
  position: relative;
  z-index: 1;
  font-size: 0.64rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #476C99;
  margin-bottom: var(--pb-sp-2);
}
.pb-hero-title {
  position: relative;
  z-index: 1;
  font-size: 1.5rem;
  font-weight: 600;
  letter-spacing: -0.025em;
  line-height: 1.25;
  color: var(--pb-text);
  margin: 0 0 var(--pb-sp-2) 0 !important;
}
.pb-hero-sub {
  position: relative;
  z-index: 1;
  color: var(--pb-text-2-strong);
  font-size: 0.82rem;
  line-height: 1.65;
  margin: 0;
  max-width: 72ch;
}

.pb-section-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--pb-text);
  margin: 0 0 var(--pb-sp-4) 0;
  padding-left: 12px;
  border-left: 3px solid var(--pb-primary-hover);
  line-height: 1.3;
}
.pb-section-sub {
  color: var(--pb-text-2-strong);
  font-size: 0.78rem;
  margin: -8px 0 var(--pb-sp-4) 24px;
}

.pb-card {
  background: var(--pb-surface);
  border: 1px solid var(--pb-border);
  border-radius: var(--pb-r-md);
  padding: var(--pb-sp-6);
  box-shadow: var(--pb-shadow);
  max-width: 100%;
  box-sizing: border-box;
}
.pb-card--tight { padding: var(--pb-sp-4); }
.pb-card + .pb-card { margin-top: var(--pb-sp-4); }

/* Native st.container(border=True) is rendered as a bordered stVerticalBlock.
   A hidden marker element scopes the styling to just those card containers
   (raw <div> wrappers are unreliable because every markdown call becomes its
   own element container in Streamlit). */
.pb-frame-marker { display: none !important; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .pb-frame-marker) {
  background: var(--pb-surface) !important;
  border: 1px solid var(--pb-border) !important;
  border-radius: var(--pb-r-md) !important;
  padding: var(--pb-sp-6) !important;
  box-shadow: var(--pb-shadow) !important;
  max-width: 100%;
}

.pb-facts {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--pb-sp-4);
}
.pb-fact {
  background: var(--pb-surface-2);
  border-radius: var(--pb-r-sm);
  padding: var(--pb-sp-4);
}
.pb-fact-label {
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 600;
  color: var(--pb-text-2-strong);
}
.pb-fact-value {
  font-size: 1.02rem;
  font-weight: 600;
  color: var(--pb-text);
  margin-top: 6px;
  letter-spacing: -0.02em;
}

.pb-kicker {
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 600;
  color: var(--pb-text-2-strong);
  margin-bottom: 6px;
}
.pb-big-value {
  font-size: 1.7rem;
  font-weight: 600;
  letter-spacing: -0.035em;
  color: var(--pb-text);
  line-height: 1.1;
}
.pb-subtle { color: var(--pb-text-2-strong); font-size: 0.78rem; line-height: 1.6; }

.pb-pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 5px 11px;
  border-radius: var(--pb-pill);
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  white-space: nowrap;
}
.pb-pill::before {
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
  flex: 0 0 6px;
}
.pb-pill--neutral { background: var(--pb-primary-light); color: #3E6288; }
.pb-pill--success { background: #EAF6F0; color: #366B53; }
.pb-pill--warning { background: #FDF4E6; color: #8A5E1A; }
.pb-pill--error   { background: #FCECEC; color: #934545; }

.pb-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: var(--pb-sp-3); }
.pb-chip {
  background: var(--pb-primary-light);
  color: #3E6288;
  border: 1px solid #D6E5F4;
  border-radius: var(--pb-r-sm);
  padding: 4px 9px;
  font-size: 0.7rem;
  font-family: 'SFMono-Regular', Consolas, monospace;
}
.pb-caption-note { color: var(--pb-text-2-strong); font-size: 0.7rem; margin-top: var(--pb-sp-4); }

/* ---- responsive ---------------------------------------------------------- */
@media (max-width: 1023px) {
  .pb-facts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .pb-hero { padding: var(--pb-sp-6); }
  .pb-hero-title { font-size: 1.35rem; }
  [data-testid="stMetric"] { padding: var(--pb-sp-3) var(--pb-sp-4) !important; }
  [data-testid="stMetricValue"] { font-size: 1.15rem !important; }
  /* Let the 5-up metric row wrap instead of truncating values. */
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
    flex-wrap: wrap !important;
    gap: var(--pb-sp-3) !important;
  }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
    flex: 1 1 calc(33.333% - 8px) !important;
    min-width: 150px !important;
  }
}
@media (max-width: 767px) {
  .pb-facts { grid-template-columns: 1fr; }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
    flex: 1 1 calc(50% - 6px) !important;
    min-width: 130px !important;
  }
  .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
  .pb-hero { padding: var(--pb-sp-4) var(--pb-sp-5); border-radius: var(--pb-r-md); }
  .pb-hero-title { font-size: 1.2rem; }
  .pb-hero-sub { font-size: 0.78rem; }
  .pb-card { padding: var(--pb-sp-4); }
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .pb-frame-marker) {
    padding: var(--pb-sp-4) !important;
  }
  [data-testid="stForm"] { padding: var(--pb-sp-4) !important; }
  .pb-big-value { font-size: 1.45rem; }
  .pb-section-sub { margin-left: 0; }
  section[data-testid="stSidebar"] { width: min(82vw, 320px) !important; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    transition-duration: 0.01ms !important;
    animation-duration: 0.01ms !important;
  }
}
</style>
"""


def inject_design_system() -> None:
    """Inject the global stylesheet. Call once, before any other UI element."""
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Presentational helpers
# --------------------------------------------------------------------------- #
def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    """Consistent hero header used at the top of every section."""
    sub = (
        f'<p class="pb-hero-sub">{escape(subtitle)}</p>' if subtitle else ""
    )
    st.markdown(
        f"""
        <div class="pb-hero">
          <div class="pb-eyebrow">{escape(eyebrow)}</div>
          <h1 class="pb-hero-title">{escape(title)}</h1>
          {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = "") -> None:
    """Section heading with a powder-blue accent rule."""
    sub = f'<p class="pb-section-sub">{escape(subtitle)}</p>' if subtitle else ""
    st.markdown(
        f'<div class="pb-section-title">{escape(title)}</div>{sub}',
        unsafe_allow_html=True,
    )


@contextmanager
def card_container():
    """Frame native Streamlit widgets/images/tables in a white card."""
    with st.container(border=True):
        st.markdown(
            '<span class="pb-frame-marker" aria-hidden="true"></span>',
            unsafe_allow_html=True,
        )
        yield


def card_html(body: str, css_class: str = "") -> None:
    """Render a self-contained HTML card (no native widgets inside)."""
    cls = f" {css_class}" if css_class else ""
    st.markdown(
        f'<div class="pb-card{cls}">{body}</div>', unsafe_allow_html=True
    )


def pill_html(text: str, tone: str = "neutral") -> str:
    """Return HTML for a small status pill (neutral/success/warning/error)."""
    valid = {"neutral", "success", "warning", "error"}
    tone = tone if tone in valid else "neutral"
    return f'<span class="pb-pill pb-pill--{tone}">{escape(text)}</span>'


def fact_grid(facts: list[tuple[str, str]]) -> str:
    """Return HTML for a responsive grid of label/value facts."""
    items = "".join(
        f'<div class="pb-fact"><div class="pb-fact-label">{escape(label)}</div>'
        f'<div class="pb-fact-value">{escape(value)}</div></div>'
        for label, value in facts
    )
    return f'<div class="pb-facts">{items}</div>'
