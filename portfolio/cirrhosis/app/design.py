"""Accessible, minimalist design system for the Streamlit app.

Goals: large readable text, generous spacing, high contrast, predictable layout,
and no reliance on colour alone to communicate meaning.
"""

from __future__ import annotations

import streamlit as st

# Muted, accessible light palette.
PALETTE = {
    "background": "#F8FAFB",
    "surface": "#FFFFFF",
    "secondary": "#EAF0F4",
    "primary": "#3D7C9B",
    "primary_soft": "#D5E6EE",
    "text": "#22303B",
    "muted": "#546575",
    "border": "#D7E0E7",
    "success": "#3F7A63",
    "warning": "#A85C3E",
}

CSS = """
<style>
  :root { color-scheme: light; }
  html { font-size: 17px; }
  body, .stApp { background: #F8FAFB; color: #22303B; font-family: -apple-system,
      BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
  .block-container { max-width: 1040px; padding-top: 1.8rem; padding-bottom: 4rem; }

  h1, h2, h3, h4 { color: #22303B; line-height: 1.3; letter-spacing: -0.01em; }
  h1 { font-size: 2rem; margin-bottom: .5rem; }
  h2 { font-size: 1.4rem; margin-top: 1.6rem; }
  h3 { font-size: 1.12rem; margin-top: 1.2rem; }
  p, li, label, .stMarkdown { font-size: 1rem; line-height: 1.65; }

  .hero { background: #FFFFFF; border: 1px solid #D7E0E7; border-radius: 14px;
      padding: 1.5rem 1.7rem; margin-bottom: 1.5rem; }
  .hero h1 { margin-top: 0; }
  .hero p { color: #546575; margin-bottom: 0; max-width: 70ch; }
  .eyebrow { color: #3D7C9B; font-size: .82rem; font-weight: 700;
      letter-spacing: .08em; text-transform: uppercase; }

  .card { background: #FFFFFF; border: 1px solid #D7E0E7; border-radius: 12px;
      padding: 1.1rem 1.2rem; }
  .card .label { font-size: .92rem; color: #546575; }
  .card .value { font-size: 1.6rem; font-weight: 700; color: #22303B; margin-top: .25rem; }
  .card .note { font-size: .88rem; color: #546575; margin-top: .3rem; }

  .note-box { background: #EAF0F4; border-left: 4px solid #3D7C9B;
      border-radius: 8px; padding: .9rem 1.1rem; margin: 1rem 0;
      font-size: .98rem; line-height: 1.6; }
  .warning-box { background: #F7EDE7; border-left: 4px solid #A85C3E;
      border-radius: 8px; padding: .9rem 1.1rem; margin: 1rem 0;
      font-size: .96rem; line-height: 1.6; color: #6B3A28; }

  .outcome { font-size: 1.5rem; font-weight: 700; color: #22303B; margin: .4rem 0; }
  .badge { display: inline-block; padding: .25rem .7rem; border-radius: 999px;
      background: #D5E6EE; color: #244F66; font-size: .9rem; font-weight: 700; }

  .prob-row { margin: 1rem 0; }
  .prob-head { display: flex; justify-content: space-between; gap: 1rem;
      font-size: 1rem; margin-bottom: .35rem; }
  .prob-name { color: #22303B; }
  .prob-value { font-weight: 700; color: #22303B; white-space: nowrap; }
  .prob-track { height: 22px; background: #EAF0F4; border-radius: 11px;
      border: 1px solid #D7E0E7; overflow: hidden; }
  .prob-fill { height: 100%; background: #3D7C9B; border-radius: 11px; }

  /* Large, obvious interactive targets. */
  .stButton > button { min-height: 52px; font-size: 1.06rem !important;
      font-weight: 700 !important; border-radius: 10px !important; }
  .stButton > button[kind="primary"] { background: #3D7C9B !important;
      color: #FFFFFF !important; }
  .stNumberInput input, .stSelectbox select { font-size: 1rem !important;
      min-height: 48px; }
  .stRadio label { font-size: 1rem !important; }
  div[data-testid="stSidebar"] { background: #EAF0F4; border-right: 1px solid #D7E0E7; }
  .stDataFrame, .stTable { font-size: .95rem; }

  @media (max-width: 820px) {
    html { font-size: 16px; }
    .block-container { padding: 1rem 1rem 3rem 1rem; }
    .hero { padding: 1.1rem 1.2rem; }
    h1 { font-size: 1.65rem; }
    .prob-head { flex-wrap: wrap; }
  }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, description: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">Cirrhosis outcome estimates</div>'
        f"<h1>{title}</h1><p>{description}</p></div>",
        unsafe_allow_html=True,
    )


def step(title: str) -> None:
    st.markdown(f"### {title}")


def note(text: str) -> None:
    st.markdown(f'<div class="note-box">{text}</div>', unsafe_allow_html=True)


def warning(text: str) -> None:
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)


def fact(label: str, value: str, note_text: str = "") -> None:
    extra = f'<div class="note">{note_text}</div>' if note_text else ""
    st.markdown(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{extra}</div>',
        unsafe_allow_html=True,
    )


def probability_row(name: str, description: str, value: float, is_top: bool) -> str:
    """Accessible horizontal bar: text + percentage, never colour alone."""
    marker = ' <span class="badge">Most likely</span>' if is_top else ""
    return (
        f'<div class="prob-row"><div class="prob-head">'
        f'<span class="prob-name">{name} — {description}{marker}</span>'
        f'<span class="prob-value">{value * 100:.1f}%</span></div>'
        f'<div class="prob-track"><div class="prob-fill" style="width:{value * 100:.1f}%"></div></div>'
        f"</div>"
    )
