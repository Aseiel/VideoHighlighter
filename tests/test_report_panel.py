"""The Report tab: which pages it finds, how a click becomes a seek, and which
times in the highlight report are marked as clickable."""
import os
import time

import pytest

from modules.report.highlight_report import build_report, render_html, source_time

try:                                    # free's CI has no PySide6
    from video_ai_editor import report_panel as rp
except ImportError:
    rp = None
needs_qt = pytest.mark.skipif(rp is None, reason="PySide6 is not installed")


@needs_qt
class TestSeekAddress:
    def test_round_trips_seconds_and_a_video_with_awkward_characters(self):
        video = r"D:\m\a b&c – ⧸.mp4"
        assert rp.parse_seek(rp.seek_url(83.5, video)) == (83.5, video)

    def test_without_a_video_it_is_the_viewers_own(self):
        assert rp.parse_seek(rp.seek_url(12)) == (12.0, None)

    @pytest.mark.parametrize("url", [
        "https://example.com/?t=3",                         # any other site
        "https://seek.videohighlighter.invalid.example.com/?t=3",   # not the host itself
        "https://seek.videohighlighter.invalid/?t=abc",
        "https://seek.videohighlighter.invalid/?t=-1",
        "https://seek.videohighlighter.invalid/?t=nan",
        "https://seek.videohighlighter.invalid/",
        "",
    ])
    def test_anything_else_is_not_a_seek(self, url):
        assert rp.parse_seek(url) is None

    def test_the_host_can_never_resolve(self):
        # .invalid is reserved (RFC 2606): a refused navigation that somehow
        # got through would still not reach the network.
        assert rp.SEEK_HOST.endswith(".invalid")


@needs_qt
class TestReportsFor:
    def _page(self, path, videos=(), age=0.0):
        meta = (f'<meta name="{rp.VIDEOS_META}" content="{"|".join(videos)}">'
                if videos else "")
        path.write_text(f"<html><head>{meta}</head><body></body></html>", encoding="utf-8")
        t = time.time() - age
        os.utime(path, (t, t))
        return str(path)

    def test_its_own_highlight_reports_and_pages_that_name_it(self, tmp_path):
        video = tmp_path / "film.mp4"
        video.write_bytes(b"")
        own = self._page(tmp_path / "film_highlight_why.html", age=10)
        both = self._page(tmp_path / "compare.html", videos=[str(video), "other.mp4"])
        self._page(tmp_path / "other_highlight_why.html")
        self._page(tmp_path / "notes.html")
        assert rp.reports_for(str(video)) == [both, own]      # newest first

    def test_a_page_about_other_videos_is_not_listed(self, tmp_path):
        video = tmp_path / "film.mp4"
        video.write_bytes(b"")
        self._page(tmp_path / "compare.html", videos=[str(tmp_path / "x.mp4")])
        assert rp.reports_for(str(video)) == []

    def test_the_meta_is_unescaped(self, tmp_path):
        page = tmp_path / "p.html"
        page.write_text(f'<meta name="{rp.VIDEOS_META}" content="D:/a &amp; b.mp4|c.mp4">',
                        encoding="utf-8")
        assert rp.videos_named_in(str(page)) == ["D:/a & b.mp4", "c.mp4"]

    def test_no_video_no_reports(self):
        assert rp.reports_for("") == []


class TestSourceTimes:
    def _report(self):
        from tests.test_highlight_report import _score, _signals
        sig = _signals(object={17: 10.0}, audio={17: 5.0})
        return build_report(video_path="a.mp4", video_duration=60, score=_score(sig),
                            signals=sig, segments=[(10, 20)],
                            object_detections={17: ["bench_vice"]},
                            settings={"clip_time": 10})

    def test_marker_carries_the_seconds_and_escapes_the_text(self):
        assert source_time("1:23 <b>", 83.456) == '<span data-vh-t="83.46">1:23 &lt;b&gt;</span>'

    def test_a_clip_card_marks_its_source_range(self):
        page = render_html(self._report())
        assert '<span class="rng"><span data-vh-t="10.00">0:10 – 0:20</span></span>' in page

    def test_times_in_the_output_are_never_marked(self):
        # The strip's tooltips give each clip's place in the rendered output;
        # a click there must not move the source playhead.
        report = self._report()
        page = render_html(report)
        out = report["segments"][0]["output_range"]
        assert f'{out} in the output' in page
        assert f'">{out}</span>' not in page


@needs_qt
class TestViewerFollowsAClick:
    """`SignalTimelineWindow._seek_from_report`, on stand-in windows."""

    @pytest.fixture
    def viewer_cls(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        stv = pytest.importorskip("signal_timeline_viewer")
        return stv.SignalTimelineWindow

    def _window(self, video, owner=None, report=None):
        import types
        w = types.SimpleNamespace(video_path=video, _owner=owner, jumps=[], shown=[])
        w._jump_to_time = w.jumps.append
        w.show_report = lambda p: w.shown.append(p) or True
        w.report_panel = types.SimpleNamespace(path=report)
        return w

    def test_a_time_in_this_video_moves_the_playhead(self, viewer_cls):
        w = self._window("a.mp4")
        viewer_cls._seek_from_report(w, 83.5, "")
        assert w.jumps == [83.5]

    def test_a_time_in_another_video_opens_it_with_the_same_report(self, viewer_cls, tmp_path):
        import types
        other = tmp_path / "b.mp4"
        other.write_bytes(b"")
        new = self._window(str(other))
        owner = types.SimpleNamespace(timeline_window=None, opened=[])

        def open_timeline_viewer(video_path=None):
            owner.opened.append(video_path)
            owner.timeline_window = new
        owner.open_timeline_viewer = open_timeline_viewer
        w = self._window("a.mp4", owner=owner, report="compare.html")
        owner.timeline_window = w
        viewer_cls._seek_from_report(w, 12.0, str(other))
        assert owner.opened == [str(other)]
        assert new.shown == ["compare.html"] and new.jumps == [12.0]
        assert w.jumps == []

    def test_a_missing_video_does_nothing(self, viewer_cls, tmp_path):
        import types
        owner = types.SimpleNamespace(open_timeline_viewer=lambda **k: pytest.fail("opened"))
        w = self._window("a.mp4", owner=owner)
        viewer_cls._seek_from_report(w, 1.0, str(tmp_path / "gone.mp4"))
        assert w.jumps == []


@needs_qt
class TestPopOut:
    """`ReportHost`: the panel moves between the Report tab and a window of its
    own, keeps the choice, and closing the window docks it back."""

    @pytest.fixture
    def qt(self, tmp_path, monkeypatch):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow
        app = QApplication.instance() or QApplication([])
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
        monkeypatch.setattr(rp, "web_engine_available", lambda: False)   # no browser needed
        made = []

        def build():
            viewer = QMainWindow()
            dock = QDockWidget("Report", viewer)
            panel = rp.ReportPanel("")
            host = rp.ReportHost(viewer, dock, panel, title="Report — film.mp4")
            viewer.show()
            made.append(viewer)
            app.processEvents()
            return viewer, dock, panel, host
        yield app, build
        for v in made:
            v.close()
            v.deleteLater()
        app.processEvents()

    def test_pop_out_moves_the_panel_into_a_real_window_and_back(self, qt):
        from PySide6.QtCore import Qt
        app, build = qt
        viewer, dock, panel, host = build()
        assert panel.window() is viewer
        panel.pop_btn.setChecked(True)
        app.processEvents()
        assert panel.window() is host.window and host.window.isVisible()
        assert host.window.windowFlags() & Qt.WindowType.Window
        assert not (host.window.windowFlags() & Qt.WindowType.Tool & ~Qt.WindowType.Window)
        assert host.window.windowTitle() == "Report — film.mp4" and dock.isHidden()
        panel.pop_btn.setChecked(False)
        app.processEvents()
        assert panel.window() is viewer and not host.window.isVisible()

    def test_closing_the_window_docks_the_report_back(self, qt):
        app, build = qt
        viewer, dock, panel, host = build()
        panel.pop_btn.setChecked(True)
        app.processEvents()
        host.window.close()
        app.processEvents()
        assert not panel.pop_btn.isChecked() and panel.window() is viewer

    def test_the_next_viewer_opens_it_where_it_was_left(self, qt):
        app, build = qt
        _v, _d, first, _h = build()
        first.pop_btn.setChecked(True)
        app.processEvents()
        _v2, _d2, second, host2 = build()
        app.processEvents()
        assert second.pop_btn.isChecked() and second.window() is host2.window

    def test_the_window_hides_and_returns_with_the_viewer(self, qt):
        app, build = qt
        viewer, _d, panel, host = build()
        panel.pop_btn.setChecked(True)
        app.processEvents()
        viewer.hide()
        app.processEvents()
        assert not host.window.isVisible()
        viewer.show()
        app.processEvents()
        assert host.window.isVisible()
