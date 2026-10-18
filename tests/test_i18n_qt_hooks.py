"""Qt boundary hooks route widget text through i18n.t()."""

from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from modules import i18n


def _app():
    return QApplication.instance() or QApplication([])


def test_qlabel_uses_catalog_when_hooks_installed():
    _app()
    i18n.set_language("tr")
    i18n.install_qt_hooks()
    label = QLabel("Download")
    assert label.text() == "İndir"


def test_set_text_also_translates():
    _app()
    i18n.set_language("de")
    i18n.install_qt_hooks()
    btn = QPushButton("Cancel")
    assert btn.text() == "Abbrechen"
    btn.setText("About")
    assert btn.text() == "Info"
