"""Shared FilmClear workspace styling with light and dark palettes."""

from string import Template

_WORKSPACE = Template("""
QMainWindow, QWidget#workspace { background: $window; color: $text; }
QWidget#sidebar { background: $panel; }
QDialog#exportDialog { background: $panel; color: $text; }
QLabel { background: transparent; color: $text; font-size: 13px; }
QLabel#titleLabel { font-size: 17px; font-weight: 650; }
QLabel#sectionLabel { font-size: 12px; font-weight: 650; color: $text; }
QLabel#subtitleLabel, QLabel#zoomLabel { font-size: 11px; color: $muted; }
QFrame#card { background: $surface; border: 1px solid $line; border-radius: 9px; }
QFrame#canvasTools { background: $panel; border: 1px solid $line; border-radius: 8px; }
QFrame#canvasToolDivider { background: $line; }
QSplitter::handle { background: $line; }
QSplitter::handle:horizontal { width: 1px; }
QGraphicsView {
    background: #222a33;
    border: 1px solid $line;
    border-radius: 8px;
}
QMenuBar, QMenu {
    background: $panel;
    color: $text;
}
QMenu { padding: 5px; border: 1px solid $line; }
QMenu::item { padding: 6px 18px; }
QMenu::item:selected { background: $selection; }
QMenu::item:disabled { color: $muted; }
QStatusBar { background: $panel; border-top: 1px solid $line; padding: 2px 6px; }
QStatusBar QLabel { padding: 0 7px; font-size: 11px; color: $muted; }
QStatusBar::item { border: none; }
QPushButton {
    background: $primary;
    color: $on_primary;
    border: 1px solid transparent;
    border-radius: 6px;
    min-height: 28px;
    padding: 2px 11px;
    font-size: 12px;
    font-weight: 650;
}
QPushButton:hover { background: $primary_hover; }
QPushButton:pressed { background: $primary_pressed; }
QPushButton:disabled { background: $disabled; color: $muted; }
QPushButton#secondaryBtn, QPushButton#toolBtn, QPushButton#canvasToolBtn {
    background: $surface;
    color: $text;
    border-color: $line;
}
QPushButton#secondaryBtn:hover, QPushButton#toolBtn:hover, QPushButton#canvasToolBtn:hover { background: $hover; }
QPushButton#secondaryBtn:disabled, QPushButton#toolBtn:disabled, QPushButton#canvasToolBtn:disabled {
    background: $panel;
    color: $muted;
}
QPushButton#toolBtn { padding: 2px 8px; min-width: 22px; }
QPushButton#canvasToolBtn { padding: 2px 3px; }
QPushButton#toolBtn:checked, QPushButton#canvasToolBtn:checked {
    background: $selection;
    border-color: $accent_line;
}
QPushButton#textBtn {
    background: transparent;
    border: none;
    color: $muted;
    padding: 2px 3px;
    min-height: 22px;
    font-size: 11px;
}
QPushButton#textBtn:hover { color: $text; background: $hover; }
QPushButton#photoFilterBtn {
    background: $surface;
    color: $muted;
    border-color: $line;
    padding: 0 2px;
    min-height: 23px;
    font-size: 11px;
}
QPushButton#photoFilterBtn:hover { background: $hover; color: $text; }
QPushButton#photoFilterBtn:checked { background: $selection; border-color: $accent_line; color: $text; }
QListWidget {
    background: $surface;
    color: $text;
    border: 1px solid $line;
    border-radius: 8px;
    padding: 3px;
    outline: none;
    font-size: 12px;
    alternate-background-color: $panel;
}
QListWidget#fileList { background: transparent; border: none; padding: 0; }
QListWidget::item { padding: 5px 6px; min-height: 22px; border-radius: 5px; }
QListWidget::item:selected { background: $selection; color: $text; }
QListWidget::item:hover:!selected { background: $hover; }
QComboBox, QSpinBox, QLineEdit {
    background: $surface;
    color: $text;
    border: 1px solid $line;
    border-radius: 6px;
    min-height: 26px;
    padding: 2px 6px;
    font-size: 12px;
}
QComboBox QAbstractItemView { background: $surface; color: $text; selection-background-color: $selection; }
QSlider::groove:horizontal { height: 3px; background: $line; border-radius: 1px; }
QSlider::handle:horizontal { width: 13px; margin: -5px 0; border-radius: 6px; background: $primary; }
QToolTip { background: $surface; color: $text; border: 1px solid $line; padding: 4px; }
QFrame#loadingOverlay { background: rgba(10, 15, 21, 154); }
QFrame#loadingCard { background: #26333f; border: 1px solid #506375; border-radius: 12px; }
QLabel#loadingLabel { color: #eef5fa; font-weight: 650; }
QProgressBar {
    background: #202b35;
    color: #eef5fa;
    border: 1px solid #506375;
    border-radius: 5px;
    text-align: center;
}
QProgressBar::chunk { background: #197565; border-radius: 4px; }
""")

LIGHT_THEME = _WORKSPACE.substitute(
    window="#edf1f4", panel="#f6f8fa", surface="#ffffff", text="#202a35",
    muted="#637182", line="#d8e0e7", hover="#eef3f6", selection="#deeee9",
    primary="#197565", primary_hover="#156455", primary_pressed="#124f45",
    on_primary="#ffffff", accent_line="#7cb8aa", disabled="#e7ecef",
)

DARK_THEME = _WORKSPACE.substitute(
    window="#151b22", panel="#202830", surface="#27313b", text="#e5edf4",
    muted="#a0afbd", line="#384552", hover="#313d49", selection="#2e4949",
    primary="#79d9c4", primary_hover="#8be5d0", primary_pressed="#5fc9b2",
    on_primary="#102a29", accent_line="#6bb8aa", disabled="#303d49",
)
