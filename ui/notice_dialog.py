"""Themed workspace confirmations and operation messages."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QLayout, QPlainTextEdit, QPushButton,
    QSizePolicy, QVBoxLayout,
)


class NoticeDialog(QDialog):
    def __init__(self, parent, title, message, *, confirm=None, cancel="取消", warning=False, details=""):
        super().__init__(parent, Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle(title)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSizeConstraint(QLayout.SetFixedSize)
        card = QFrame()
        card.setObjectName("noticeCard")
        card.setFixedWidth(436)
        layout.addWidget(card)
        content = QVBoxLayout(card)
        content.setContentsMargins(20, 20, 20, 16)
        content.setSpacing(16)

        header = QHBoxLayout()
        header.setSpacing(12)
        warning = warning or confirm is not None
        badge = QLabel("!" if warning else "i")
        badge.setObjectName("noticeWarningIcon" if warning else "noticeInfoIcon")
        badge.setFixedSize(32, 32)
        badge.setAlignment(Qt.AlignCenter)
        header.addWidget(badge, alignment=Qt.AlignTop)

        text = QVBoxLayout()
        text.setSpacing(6)
        heading = QLabel(title)
        heading.setObjectName("noticeTitle")
        text.addWidget(heading)
        self.message_label = QLabel(message)
        self.message_label.setTextFormat(Qt.PlainText)
        self.message_label.setWordWrap(True)
        self.message_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        text.addWidget(self.message_label)
        header.addLayout(text, 1)
        content.addLayout(header)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        if details:
            editor = QPlainTextEdit(details)
            editor.setObjectName("noticeDetails")
            editor.setReadOnly(True)
            editor.setFixedHeight(126)
            editor.hide()
            content.addWidget(editor)
            toggle = QPushButton("查看详情")
            toggle.setObjectName("textBtn")
            toggle.setAutoDefault(False)
            toggle.setCheckable(True)
            toggle.toggled.connect(editor.setVisible)
            toggle.toggled.connect(lambda shown: toggle.setText("收起详情" if shown else "查看详情"))
            actions.addWidget(toggle)
        actions.addStretch()

        self.cancel_button = None
        if confirm is not None:
            self.cancel_button = QPushButton(cancel)
            self.cancel_button.setObjectName("secondaryBtn")
            self.cancel_button.clicked.connect(self.reject)
            actions.addWidget(self.cancel_button)
        self.confirm_button = QPushButton(confirm or "关闭")
        if confirm is not None:
            self.confirm_button.setObjectName("noticeDangerBtn")
            self.confirm_button.setAutoDefault(False)
        self.confirm_button.clicked.connect(self.accept)
        actions.addWidget(self.confirm_button)
        content.addLayout(actions)

        default = self.cancel_button or self.confirm_button
        default.setDefault(True)
        default.setFocus()
