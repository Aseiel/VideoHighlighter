"""Central UI theme — one place for colour and a global stylesheet.

The app grew a violet cast because every widget carried its own inline
`setStyleSheet` with hardcoded hex; there was no single source of truth, so
retheming meant hunting ~100 literals across a dozen files. This module is that
source of truth: a small set of design tokens plus one Qt stylesheet applied
once at startup (`apply(app)`), so the base look of every window, card, button
and field is defined here and nowhere else.

It is intentionally additive. Existing inline styles still win over the global
sheet (Qt resolves the most specific rule), so dropping this in changes nothing
that's already styled and only fills in the unstyled defaults — the window
chrome, group boxes, plain buttons, inputs, scrollbars. Widgets can then shed
their inline styles over time and inherit from here instead.

Kept in its own module so the shared UI files stay byte-identical across the two
repos: the whole delta is this file plus a one-line call in main.

Dark only for now; `Palette` is a dataclass so a light variant is a second
instance and a toggle later, not a rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    """Every colour the stylesheet needs, named by role rather than by hue so a
    variant (light mode) only swaps values, never selectors."""

    # Surfaces, darkest (page) to lightest (elevated control)
    bg: str            # window / page canvas
    surface: str       # cards, group boxes
    surface_alt: str   # inputs, list rows
    surface_hi: str    # hover / elevated

    # Lines
    border: str
    border_strong: str

    # Text
    text: str
    text_dim: str
    text_mute: str

    # The single accent + its states, and what text sits on it
    accent: str
    accent_hover: str
    accent_press: str
    on_accent: str

    # Semantic (status), left saturated on purpose
    danger: str
    success: str
    warning: str

    # Geometry
    radius: int        # controls
    radius_card: int   # cards


# Graphite + one crisp blue — the look the inline de-blueing already converged on.
DARK = Palette(
    bg="#141414",
    surface="#1c1c1c",
    surface_alt="#242424",
    surface_hi="#2c2c2c",
    border="#2e2e2e",
    border_strong="#3d3d3d",
    text="#e6e6e6",
    text_dim="#a0a0a0",
    text_mute="#6e6e6e",
    accent="#2f81f7",
    accent_hover="#4a90f5",
    accent_press="#1f6fe0",
    on_accent="#ffffff",
    danger="#e5484d",
    success="#3fb950",
    warning="#f0883e",
    radius=8,
    radius_card=12,
)


def build_qss(p: Palette = DARK) -> str:
    """The global Qt stylesheet for a palette.

    Scoped to base widget types only. Anything a screen needs to look special
    (a coloured action button, the timeline's painted canvas) keeps its own
    inline style and overrides this.
    """
    return f"""
    QWidget {{
        background-color: {p.bg};
        color: {p.text};
        font-family: "Segoe UI", "Inter", Arial, sans-serif;
        font-size: 10pt;
    }}
    QMainWindow, QDialog {{ background-color: {p.bg}; }}

    QToolTip {{
        background-color: {p.surface_hi};
        color: {p.text};
        border: 1px solid {p.border_strong};
        border-radius: {p.radius}px;
        padding: 4px 8px;
    }}

    /* Cards — outlined, not filled. A filled card (surface over the #141414
       page) leaves a lighter frame in the padding while the content area stays
       page-dark, which reads as a confusing inner shade band. Transparent fill
       + border keeps everything one colour; the border alone defines the card. */
    QGroupBox {{
        background-color: transparent;
        border: 1px solid {p.border_strong};
        border-radius: {p.radius_card}px;
        margin-top: 14px;
        padding: 14px 12px 12px 12px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 4px;
        color: {p.text_dim};
    }}

    /* Default (secondary) button — quiet, bordered. Coloured CTAs style
       themselves inline and keep their colour. */
    QPushButton {{
        background-color: {p.surface_alt};
        color: {p.text};
        border: 1px solid {p.border_strong};
        border-radius: {p.radius}px;
        padding: 6px 14px;
    }}
    QPushButton:hover {{ background-color: {p.surface_hi}; border-color: {p.accent}; }}
    QPushButton:pressed {{ background-color: {p.bg}; }}
    QPushButton:disabled {{ color: {p.text_mute}; border-color: {p.border}; background-color: {p.surface}; }}

    /* Text-ish inputs. Spin boxes are left to Fusion + palette (see the note by
       the combo rules) so their native arrows survive. */
    QLineEdit, QPlainTextEdit, QTextEdit, QComboBox {{
        background-color: {p.surface_alt};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: {p.radius}px;
        padding: 5px 8px;
        selection-background-color: {p.accent};
        selection-color: {p.on_accent};
    }}
    QLineEdit:hover, QComboBox:hover {{ border-color: {p.border_strong}; }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{ border-color: {p.accent}; }}

    /* Disabled state. Hardcoding colour/background above overrides the palette's
       own disabled dimming, so a disabled combo would look identical to a live
       one — the sections that used to grey out (transcript rows behind their
       checkbox) stopped reading as off. Restore the dim explicitly. */
    QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled, QComboBox:disabled {{
        color: {p.text_mute};
        background-color: {p.surface};
        border-color: {p.border};
    }}
    QLabel:disabled, QCheckBox:disabled, QRadioButton:disabled {{ color: {p.text_mute}; }}

    /* Spin boxes are intentionally NOT restyled here: partial QSS on them makes
       Qt drop the native arrows for blank boxes. Under Fusion + the dark palette
       (see apply) their native up/down arrows render clean and dark on their
       own. The one safe rule is a width cap — they hold 1-5 digit values, so
       stretching them across a form column is wasted space. (max-width is a box
       property; it doesn't disturb the native arrows.) */
    QSpinBox, QDoubleSpinBox {{ max-width: 100px; }}

    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox::down-arrow {{
        width: 0; height: 0; border-left: 4px solid transparent;
        border-right: 4px solid transparent; border-top: 5px solid {p.text_dim};
        margin-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {p.surface_alt};
        color: {p.text};
        border: 1px solid {p.border_strong};
        selection-background-color: {p.accent};
        selection-color: {p.on_accent};
        outline: none;
    }}

    /* Checkboxes / radios — accent when on */
    QCheckBox, QRadioButton {{ spacing: 7px; background: transparent; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px; height: 16px;
        border: 1px solid {p.border_strong};
        background-color: {p.surface_alt};
    }}
    QCheckBox::indicator {{ border-radius: 4px; }}
    QRadioButton::indicator {{ border-radius: 8px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {p.accent}; }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {p.accent}; border-color: {p.accent};
    }}

    /* Tabs */
    QTabWidget::pane {{ border: 1px solid {p.border}; border-radius: {p.radius}px; top: -1px; }}
    QTabBar::tab {{
        background: transparent;
        color: {p.text_dim};
        padding: 7px 14px;
        border: 1px solid transparent;
        border-top-left-radius: {p.radius}px;
        border-top-right-radius: {p.radius}px;
    }}
    QTabBar::tab:hover {{ color: {p.text}; }}
    QTabBar::tab:selected {{
        color: {p.text};
        background: {p.surface};
        border-color: {p.border};
        border-bottom-color: {p.surface};
    }}

    /* Sliders */
    QSlider::groove:horizontal {{ height: 4px; background: {p.border_strong}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {p.accent}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: {p.accent}; width: 16px; margin: -7px 0; border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{ background: {p.accent_hover}; }}

    /* Progress */
    QProgressBar {{
        background-color: {p.surface_alt};
        border: none; border-radius: {p.radius}px;
        text-align: center; color: {p.text};
    }}
    QProgressBar::chunk {{ background-color: {p.accent}; border-radius: {p.radius}px; }}

    /* Scrollbars — thin, no arrows */
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {p.border_strong}; border-radius: 5px; min-height: 24px; }}
    QScrollBar::handle:vertical:hover {{ background: {p.text_mute}; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {p.border_strong}; border-radius: 5px; min-width: 24px; }}
    QScrollBar::handle:horizontal:hover {{ background: {p.text_mute}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    /* Menus */
    QMenu {{
        background-color: {p.surface_alt};
        border: 1px solid {p.border_strong};
        border-radius: {p.radius}px;
        padding: 4px;
    }}
    QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
    QMenu::item:selected {{ background-color: {p.accent}; color: {p.on_accent}; }}
    QMenu::separator {{ height: 1px; background: {p.border}; margin: 4px 8px; }}
    """


def _dark_qpalette(p: Palette):
    """A dark QPalette so Fusion renders native sub-controls (spin-box arrows,
    check indicators) dark instead of on its default light base. QSS handles the
    polish on top; the palette handles the bits QSS shouldn't touch."""
    from PySide6.QtGui import QColor, QPalette

    def c(h):
        return QColor(h)

    pal = QPalette()
    pal.setColor(QPalette.Window, c(p.bg))
    pal.setColor(QPalette.WindowText, c(p.text))
    pal.setColor(QPalette.Base, c(p.surface_alt))
    pal.setColor(QPalette.AlternateBase, c(p.surface))
    pal.setColor(QPalette.ToolTipBase, c(p.surface_hi))
    pal.setColor(QPalette.ToolTipText, c(p.text))
    pal.setColor(QPalette.Text, c(p.text))
    pal.setColor(QPalette.Button, c(p.surface_alt))
    pal.setColor(QPalette.ButtonText, c(p.text))
    pal.setColor(QPalette.BrightText, c("#ffffff"))
    pal.setColor(QPalette.Highlight, c(p.accent))
    pal.setColor(QPalette.HighlightedText, c(p.on_accent))
    pal.setColor(QPalette.PlaceholderText, c(p.text_mute))
    pal.setColor(QPalette.Link, c(p.accent))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, c(p.text_mute))
    return pal


def apply(app, palette: Palette = DARK) -> None:
    """Install the theme on a QApplication. Call once, right after it's built.

    Fusion + a dark palette first: the native Windows style paints its own
    spin-box steppers (the light 3D boxes) and ignores the stylesheet's
    sub-control rules. Fusion honours both the palette and QSS, so spin boxes
    get clean native dark arrows while the rest takes our stylesheet.
    """
    try:
        app.setStyle("Fusion")
        app.setPalette(_dark_qpalette(palette))
    except Exception:
        pass
    app.setStyleSheet(build_qss(palette))
