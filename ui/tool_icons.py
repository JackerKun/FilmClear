"""Small line icons for the canvas toolbar."""

from PySide6.QtCore import QByteArray
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


_MARQUEE = '<rect x="3" y="3" width="14" height="14" rx="2" stroke-dasharray="2.5 2.5"/>'
_SHAPES = {
    "pan": '<path d="M10 2v16M2 10h16M8 4l2-2 2 2M8 16l2 2 2-2M4 8l-2 2 2 2m12-4 2 2-2 2"/>',
    "detect": _MARQUEE + '<circle cx="9" cy="9" r="3"/><path d="m11.2 11.2 2.5 2.5"/>',
    "draw": '<path d="m4 13 9-9a2 2 0 0 1 3 3l-9 9-4 1 1-4Zm7.5-7.5 3 3M3 17h14"/>',
    "check": _MARQUEE + '<path d="m6.5 10 2.3 2.3 4.7-4.7"/>',
    "ignore": _MARQUEE + '<path d="m7 7 6 6m0-6-6 6"/>',
    "repair": _MARQUEE + '<path d="m6 14 6-6m1-3v4m-2-2h4M7 7v2M6 8h2"/>',
    "marks": '<path d="M2 10s3-5 8-5 8 5 8 5-3 5-8 5-8-5-8-5Z"/><circle cx="10" cy="10" r="2"/>',
    "fit": '<path d="M7 3H3v4m10-4h4v4M3 13v4h4m10-4v4h-4"/><rect x="6" y="7" width="8" height="6" rx="1"/>',
    "zoom-out": '<circle cx="8.5" cy="8.5" r="5"/><path d="m12 12 4 4M6 8.5h5"/>',
    "zoom-in": '<circle cx="8.5" cy="8.5" r="5"/><path d="m12 12 4 4M6 8.5h5M8.5 6v5"/>',
    "wipe": '<rect x="2" y="4" width="16" height="12" rx="2"/><path d="M10 2v16m-3-9-1 1 1 1m6-2 1 1-1 1"/>',
    "side": '<rect x="2" y="4" width="6" height="12" rx="1"/><rect x="12" y="4" width="6" height="12" rx="1"/>',
    "stack": '<rect x="3" y="2" width="14" height="6" rx="1"/><rect x="3" y="12" width="14" height="6" rx="1"/>',
}


def tool_icon(name, color):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
        f'<g fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round">{_SHAPES[name]}</g></svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(32, 32)
    pixmap.fill("transparent")
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QIcon(pixmap)
