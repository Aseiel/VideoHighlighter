# Reports beside the timeline

A report answers "why"; the footage answers "is that true". The **Report** tab
puts the two side by side: any HTML report about a video opens next to its
timeline, and every time the report names is a link that moves the playhead
there. Comparing two videos is the same thing with two tabs.

Shared by both editions, as explanation is never a paid feature. The live-link
and MCP command `show_report`, which lets an assistant open a report for you,
is Pro.

## Using it

- **Report tab.** It sits in the viewer's right column with Controls, Search,
  Transcript and Narration. Its picker lists the reports found beside the
  video: the highlight report (`<video>…_why.html`) and any page that names the
  video (see *Pages about several videos*). **Open…** shows any other `.html`,
  and **In browser** opens the page in the web browser.
- **Click a time.** The playhead moves there. A time in another video switches
  to that video's tab (opening it if needed), brings the report along and
  jumps to the moment.
- **The time clicked last glows** in the highlighter yellow. It stays lit when
  the click switched videos, and the page scrolls back to it.
- **Pop out.** The report moves into a window of its own, which Windows can
  snap to a screen edge (drag it to the edge, or Win+←). A floating dock
  can't do that, because Qt makes it a tool window. The app remembers the
  choice and where the window was left. **Dock**, or closing the window, puts
  it back in the tab.
- **One viewer window, a tab per video.** Opening a video that already has a
  tab brings that tab forward without rebuilding it. Leaving a tab pauses it.
  Keys (Space, Delete, Ctrl+Z) act on the tab on screen only. Closing a tab
  pauses and mutes that video and keeps it in memory, as closing the viewer
  always did, so reopening it is quick.

## Writing a report the panel can use

Plain HTML. Nothing in it needs the app, and in a browser the marks are inert.

| Mark | Meaning |
|---|---|
| `<span data-vh-t="83.5">1:23</span>` | a moment of the source video, in seconds |
| `data-vh-video="D:\path\other.mp4"` on the same element | the moment is in that video, not the one the viewer shows |
| `<meta name="videohighlighter-videos" content="a.mp4\|b.mp4">` in `<head>` | the page is about these videos; it is listed in each one's Report tab |
| `<video src="clip.mp4#t=4,9">` | gets a **Show in timeline** button for the fragment's start |

Mark **source** positions only. The highlight report also prints times inside
the rendered output, and those are deliberately left unmarked: guessing which
clock a bare `m:ss` belongs to would jump to the wrong place. The highlight
report's own player buttons (`button.seek`, `button.qseek`) carry source
seconds, so the panel takes those over too.

## Pages about several videos

A comparison of two (or more) videos is an ordinary page that lists them in the
`videohighlighter-videos` meta and marks each time with `data-vh-video`. Opened
from either video, it shows in that video's Report tab. Clicks move between the
videos' tabs, so a claim about one video can be checked against the other in a
click each.

## How it works

`video_ai_editor/report_panel.py` (Qt WebEngine) injects one script when the
page loads. A click on a marked time becomes a navigation to
`https://seek.videohighlighter.invalid/?t=…&video=…`, which the page object
refuses and turns into a seek. `.invalid` is reserved and never resolves, so
nothing leaves the machine. Other links open in the web browser and are never
followed inside the panel. `video_ai_editor/viewer_tabs.py` holds the viewers
as tabs, and the main window keeps one viewer per video.

Release builds bundle Qt WebEngine's widgets and its browser process. The
smoke test fails a build without the process, and `THIRD-PARTY-NOTICES.md`
covers the licence. Without WebEngine (a stripped environment) the tab says
so and offers **In browser**.

For an assistant: `show_report(video_path, report_path, popped=None)` in the
MCP server (Pro, needs the live link).
