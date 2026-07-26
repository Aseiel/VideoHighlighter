// A global gate limiting how many video posters decode at once.
//
// Each VideoCard spins up a hidden <video> to grab a poster frame. With a whole
// folder of clips added at once (a GoPro dump is easily 150+ files), mounting
// them all together opens that many decoders and native buffers simultaneously,
// which exhausts WebView2 and looks like the app crashing. This caps concurrent
// decodes so the rest queue and run as slots free.

const MAX_CONCURRENT = 4

let active = 0
const waiting: Array<() => void> = []

/** Wait for a decode slot. Returns a release() to call when the poster is done
 *  (captured or failed) — always call it, or the queue stalls. */
export function acquireDecodeSlot(): Promise<() => void> {
  return new Promise((resolve) => {
    const grant = () => {
      active++
      let released = false
      resolve(() => {
        if (released) return
        released = true
        active--
        const next = waiting.shift()
        if (next) next()
      })
    }
    if (active < MAX_CONCURRENT) grant()
    else waiting.push(grant)
  })
}
