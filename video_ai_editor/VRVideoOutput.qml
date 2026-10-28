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

    // Right-click and draw-a-box, so the plain surface offers what the Live
    // overlay's right-click does without the overlay switched on. Coordinates
    // reach Python as fractions of the *shown* picture (eyeClip); Python maps
    // them onto the full frame, since only it knows what VR mode crops away.
    signal contextRequested(real x, real y)                // root coords
    signal regionDrawn(real x, real y, real w, real h)     // fractions of eyeClip

    // Armed from Python (begin_region_draw): the next left-drag draws the box.
    property bool drawing: false

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

    Rectangle {
        id: band
        visible: false
        color: "#2878e68c"
        border.color: "#78e68c"
        border.width: 2
    }

    // Takes the right button only, until a box is armed: a left click still
    // goes wherever it went before.
    MouseArea {
        id: mouse
        anchors.fill: parent
        acceptedButtons: root.drawing ? (Qt.LeftButton | Qt.RightButton) : Qt.RightButton
        cursorShape: root.drawing ? Qt.CrossCursor : Qt.ArrowCursor
        focus: root.drawing
        property real ox: 0
        property real oy: 0

        function cancel() {
            root.drawing = false
            band.visible = false
        }

        onPressed: (event) => {
            if (event.button === Qt.RightButton) {
                if (root.drawing)
                    cancel()
                else
                    root.contextRequested(event.x, event.y)
                return
            }
            ox = event.x
            oy = event.y
            band.x = ox; band.y = oy; band.width = 0; band.height = 0
            band.visible = true
        }
        onPositionChanged: (event) => {
            if (!band.visible)
                return
            band.x = Math.min(ox, event.x)
            band.y = Math.min(oy, event.y)
            band.width = Math.abs(event.x - ox)
            band.height = Math.abs(event.y - oy)
        }
        onReleased: (event) => {
            if (event.button !== Qt.LeftButton || !band.visible)
                return
            // Keep the part of the box that lies on the picture, not the bars.
            const x0 = Math.max(band.x, eyeClip.x)
            const y0 = Math.max(band.y, eyeClip.y)
            const x1 = Math.min(band.x + band.width, eyeClip.x + eyeClip.width)
            const y1 = Math.min(band.y + band.height, eyeClip.y + eyeClip.height)
            cancel()
            if (x1 - x0 < 5 || y1 - y0 < 5)
                return
            root.regionDrawn((x0 - eyeClip.x) / eyeClip.width,
                             (y0 - eyeClip.y) / eyeClip.height,
                             (x1 - x0) / eyeClip.width,
                             (y1 - y0) / eyeClip.height)
        }
        Keys.onEscapePressed: cancel()
    }
}
