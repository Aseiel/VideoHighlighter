import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { browseListing, type ListingEntry } from "@/lib/api"

interface Props {
  open: boolean
  onOpenChange: (v: boolean) => void
  url: string
  /** Called with the exact URLs to fetch (no listing scrape downstream). */
  onPick: (urls: string[]) => void
}

/** Web equivalent of the Qt "Browse & Select…" thumbnail grid. */
export function VideoPickerDialog({ open, onOpenChange, url, onPick }: Props) {
  const [entries, setEntries] = useState<ListingEntry[]>([])
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState("")

  useEffect(() => {
    if (!open) return
    setEntries([])
    setPicked(new Set())
    setStatus("Loading listing…")
    setLoading(true)
    void browseListing(url).then((r) => {
      setLoading(false)
      if (!r.ok) return setStatus(`Failed to load listing: ${r.error}`)
      setEntries(r.entries)
      setStatus(r.entries.length ? "" : "No videos found on that page.")
    })
  }, [open, url])

  const toggle = (u: string) =>
    setPicked((s) => {
      const next = new Set(s)
      next.has(u) ? next.delete(u) : next.add(u)
      return next
    })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle>Select videos to download</DialogTitle>
        </DialogHeader>

        {status && (
          <p className="text-sm italic text-primary">{status}</p>
        )}

        <ScrollArea className="h-[420px] pr-3">
          <div className="grid grid-cols-3 gap-3">
            {entries.map((e) => (
              <button
                key={e.url}
                onClick={() => toggle(e.url)}
                className={`overflow-hidden rounded-md border text-left transition ${
                  picked.has(e.url) ? "ring-2 ring-primary" : "hover:bg-muted/50"
                }`}
              >
                <div className="relative grid h-28 place-items-center bg-muted">
                  {e.thumbnail_url ? (
                    <img
                      src={e.thumbnail_url}
                      alt=""
                      className="size-full object-cover"
                      loading="lazy"
                    />
                  ) : (
                    <span className="text-xs text-muted-foreground">no preview</span>
                  )}
                  {e.duration && (
                    <span className="absolute right-1 top-1 rounded bg-black/70 px-1 text-[10px] text-white">
                      {e.duration}
                    </span>
                  )}
                  <span className="absolute left-1 top-1">
                    <Checkbox checked={picked.has(e.url)} />
                  </span>
                </div>
                <p className="line-clamp-2 p-2 text-xs" title={e.title || e.url}>
                  {e.title || e.url}
                </p>
              </button>
            ))}
          </div>
        </ScrollArea>

        <DialogFooter className="justify-between sm:justify-between">
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setPicked(new Set(entries.map((e) => e.url)))}
              disabled={!entries.length}
            >
              Select all
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setPicked(new Set())}
              disabled={!picked.size}
            >
              Select none
            </Button>
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button
              disabled={!picked.size || loading}
              onClick={() => {
                onPick([...picked])
                onOpenChange(false)
              }}
              className="bg-[color:var(--success)] text-white hover:opacity-90"
            >
              Download selected ({picked.size})
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
