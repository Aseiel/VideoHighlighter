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
        w = types.SimpleNamespace(video_path=video, _owner=owner, jumps=[], shown=[],
                                  isVisible=lambda: True)
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

    def test_this_videos_time_from_a_hidden_tab_brings_the_tab_back(self, viewer_cls, tmp_path):
        import types
        me = tmp_path / "a.mp4"
        me.write_bytes(b"")
        owner = types.SimpleNamespace(timeline_window=None, opened=[])
        w = self._window(str(me), owner=owner, report="r.html")
        w.isVisible = lambda: False

        def open_timeline_viewer(video_path=None):
            owner.opened.append(video_path)
            owner.timeline_window = w
        owner.open_timeline_viewer = open_timeline_viewer
        viewer_cls._seek_from_report(w, 30.0, "")
        assert owner.opened == [str(me)] and w.jumps == [30.0] and w.shown == []

    def test_a_popped_report_is_left_alone_when_the_video_changes(self, viewer_cls, tmp_path):
        import types
        other = tmp_path / "b.mp4"
        other.write_bytes(b"")
        new = self._window(str(other))
        owner = types.SimpleNamespace(timeline_window=None)
        owner.open_timeline_viewer = lambda video_path=None: setattr(owner, "timeline_window", new)
        w = self._window("a.mp4", owner=owner, report="compare.html")
        kept = []
        w._report_host = types.SimpleNamespace(popped=True, keep_on_top=lambda: kept.append(1))
        viewer_cls._seek_from_report(w, 12.0, str(other))
        assert new.jumps == [12.0] and new.shown == [] and kept == [1]

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
        if rp._SHARED["window"] is not None:
            rp._SHARED["window"].close_quietly()
            rp._SHARED["window"].deleteLater()
        rp._SHARED.update(window=None, host=None)
        app.processEvents()

    def test_pop_out_moves_the_panel_into_a_real_window_and_back(self, qt):
        from PySide6.QtCore import Qt
        app, build = qt
        viewer, dock, panel, host = build()
        assert panel.window() is viewer
        panel.pop_btn.setChecked(True)
        app.processEvents()
        assert panel.window() is rp._SHARED["window"] and host.window.isVisible()
        assert host.window.windowFlags() & Qt.WindowType.Window
        assert not (host.window.windowFlags() & Qt.WindowType.Tool & ~Qt.WindowType.Window)
        assert host.window.windowTitle() == "Report — film.mp4" and dock.isHidden()
        panel.pop_btn.setChecked(False)
        app.processEvents()
        assert panel.window() is viewer and not rp._SHARED["window"].isVisible()

    def test_closing_the_window_docks_the_report_back(self, qt):
        app, build = qt
        viewer, dock, panel, host = build()
        panel.pop_btn.setChecked(True)
        app.processEvents()
        host.window.close()
        app.processEvents()
        assert not panel.pop_btn.isChecked() and panel.window() is viewer

    def test_a_new_viewer_does_not_take_the_window_in_use(self, qt):
        # Opening another video from a report must not swap the report out.
        app, build = qt
        _v, _d, first, _h = build()
        first.pop_btn.setChecked(True)
        app.processEvents()
        _v2, _d2, second, host2 = build()
        app.processEvents()
        assert first.pop_btn.isChecked() and not second.pop_btn.isChecked()
        assert first.window() is rp._SHARED["window"] and second.window() is _v2

    def test_a_new_viewer_pops_out_by_itself_when_the_window_is_free(self, qt):
        app, build = qt
        _v, _d, first, _h = build()
        first.pop_btn.setChecked(True)
        app.processEvents()
        first.pop_btn.setChecked(False)          # docked by hand: the choice is now "tab"
        rp.QSettings().setValue(rp.POPPED_KEY, True)
        app.processEvents()
        _v2, _d2, second, _h2 = build()
        app.processEvents()
        assert second.pop_btn.isChecked() and second.window() is rp._SHARED["window"]

    def test_popping_another_report_sends_the_first_home(self, qt):
        app, build = qt
        v1, d1, first, h1 = build()
        v2, d2, second, h2 = build()
        first.pop_btn.setChecked(True)
        second.pop_btn.setChecked(True)
        app.processEvents()
        assert second.window() is rp._SHARED["window"] and first.window() is v1
        assert not first.pop_btn.isChecked() and not d1.isHidden()

    def test_a_tab_switch_leaves_the_report_window_alone(self, qt):
        # A click on another video's time hides this viewer (its tab goes to
        # the back); the one report window must stay exactly where it is.
        app, build = qt
        viewer, _d, panel, host = build()
        panel.pop_btn.setChecked(True)
        app.processEvents()
        win = rp._SHARED["window"]
        viewer.hide()
        app.processEvents()
        assert win.isVisible() and panel.window() is win and panel.pop_btn.isChecked()

    def test_closing_the_viewer_takes_its_report_home_but_keeps_the_choice(self, qt):
        app, build = qt
        viewer, _d, panel, host = build()
        panel.pop_btn.setChecked(True)
        app.processEvents()
        viewer.close()
        app.processEvents()
        assert not rp._SHARED["window"].isVisible() and rp._SHARED["host"] is None
        assert rp._setting_bool(rp.POPPED_KEY)
