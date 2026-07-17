import QtQuick
import QtMultimedia

// Single GPU video surface used for BOTH normal and VR (SBS) playback.
//
// `vrMode` toggles between full-frame and a left-eye crop *on the same surface*,
// so switching VR on/off never creates or destroys a swapchain. That matters
// because RTSS/MSI-Afterburner paints its OSD onto every live D3D swapchain, and
// a QQuickWidget does not release its swapchain promptly — so a create/destroy
// (or two coexisting surfaces) makes the OSD appear twice. One persistent
// surface == one swapchain == one OSD.
//
// The crop trick: the VideoOutput is stretched to 2x the eye width so the full
// SBS frame maps across it; only the left half sits inside the clipped `eyeClip`
// item, so that's all that's drawn. In full-frame mode the VideoOutput is the
// eye-clip width and the whole frame shows. GPU scene-graph clip, no CPU work.
Rectangle {
    id: root
    color: "black"

    // Full-frame resolution (both eyes for SBS); set from Python via metadata.
    property real nativeWidth: 3840
    property real nativeHeight: 1080

    // true  -> crop to the left eye (side-by-side VR/3D)
    // false -> show the whole frame (normal playback)
    property bool vrMode: false

    // Exposed to Python: pass this item to QMediaPlayer.setVideoOutput(), which
    // extracts its video sink internally (PySide can't marshal QVideoSink*, but
    // a QQuickItem is a plain QObject and converts fine).
    property alias videoOutputItem: videoOut

    // Aspect of the *shown* content: half-width in VR (one eye), full otherwise.
    readonly property real contentAspect: nativeHeight > 0
        ? (root.vrMode ? (nativeWidth / 2) : nativeWidth) / nativeHeight
        : (16 / 9)
    readonly property bool fitByHeight: (root.width / Math.max(1, root.height)) > contentAspect

    Item {
        id: eyeClip
        clip: true
        anchors.centerIn: parent
        // Fit the shown content into the view, keeping aspect (letterbox/pillarbox).
        width: root.fitByHeight ? root.height * root.contentAspect : root.width
        height: root.fitByHeight ? root.height : root.width / root.contentAspect

        VideoOutput {
            id: videoOut
            objectName: "vrVideoOut"   // fetched from Python via findChild
            x: 0
            y: 0
            // VR: stretch to 2x eye width so the left half fills the clip.
            // Normal: match the clip width so the whole frame is shown.
            width: root.vrMode ? eyeClip.width * 2 : eyeClip.width
            height: eyeClip.height
            fillMode: VideoOutput.Stretch
        }
    }
}
