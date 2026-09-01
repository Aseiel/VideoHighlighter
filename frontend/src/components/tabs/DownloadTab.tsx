import { useState } from "react"
import { Download, FolderOpen, LayoutGrid } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { NumberField } from "@/components/NumberField"
import { pickDirectory, isTauri } from "@/lib/files"
import { VideoPickerDialog } from "@/components/VideoPickerDialog"
import { toast } from "sonner"

export interface DownloadSettings {
  url: string
  saveDir: string
  downloadFull: boolean
  rangeStart: number
  rangeEnd: number
  concurrent: number
  autoAdd: boolean
}

export const DEFAULT_DOWNLOAD: DownloadSettings = {
  url: "",
  saveDir: "",
  downloadFull: true,
  rangeStart: 0,
  rangeEnd: 300,
  concurrent: 1,
  autoAdd: true,
}

interface Props {
  settings: DownloadSettings
  onChange: (s: DownloadSettings) => void
  onDownload: () => void
  /** Download only these exact URLs, picked from the listing grid. */
  onDownloadUrls: (urls: string[]) => void
  running: boolean
}

const fmt = (s: number) =>
  `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`

export function DownloadTab({
  settings,
  onChange,
  onDownload,
  onDownloadUrls,
  running,
}: Props) {
  const [picking, setPicking] = useState(false)
  const [pickerOpen, setPickerOpen] = useState(false)
  const set = <K extends keyof DownloadSettings>(
    k: K,
    v: DownloadSettings[K],
  ) => onChange({ ...settings, [k]: v })

  const chooseDir = async () => {
    setPicking(true)
    const dir = await pickDirectory()
    setPicking(false)
    if (dir) set("saveDir", dir)
  }

  const duration = Math.max(0, settings.rangeEnd - settings.rangeStart)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <Download className="size-4" /> Download Videos
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid min-w-0 grid-cols-[5.5rem_minmax(0,1fr)] items-center gap-3">
          <Label className="text-sm text-muted-foreground">Page URL</Label>
          <Input
            value={settings.url}
            onChange={(e) => set("url", e.target.value)}
            placeholder="https://example.com/videos"
            className="h-8 w-full"
          />
        </div>

        <div className="grid min-w-0 grid-cols-[5.5rem_minmax(0,1fr)_auto] items-center gap-3">
          <Label className="text-sm text-muted-foreground">Save to</Label>
          <Input
            value={settings.saveDir}
            onChange={(e) => set("saveDir", e.target.value)}
            placeholder={isTauri() ? "Choose a folder…" : "D:\\movies"}
            className="h-8 w-full"
          />
          <Button
            size="sm"
            variant="secondary"
            onClick={chooseDir}
            disabled={picking}
          >
            <FolderOpen className="size-4" /> Browse
          </Button>
        </div>

        <div className="space-y-3 rounded-md border p-3">
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={settings.downloadFull}
              onCheckedChange={(v) => set("downloadFull", Boolean(v))}
            />
            Download full videos
          </label>
          {!settings.downloadFull && (
            <div className="space-y-2.5">
              <NumberField
                label="Start"
                hint="(s)"
                value={settings.rangeStart}
                onChange={(v) => set("rangeStart", v)}
              />
              <NumberField
                label="End"
                hint="(s)"
                value={settings.rangeEnd}
                min={1}
                onChange={(v) => set("rangeEnd", v)}
              />
              <p className="text-xs text-muted-foreground">
                Duration: {duration}s ({fmt(duration)})
              </p>
            </div>
          )}
        </div>

        <NumberField
          label="Concurrent downloads"
          value={settings.concurrent}
          min={1}
          onChange={(v) => set("concurrent", v)}
        />

        <label className="flex items-center gap-2 text-sm">
          <Checkbox
            checked={settings.autoAdd}
            onCheckedChange={(v) => set("autoAdd", Boolean(v))}
          />
          Add downloaded videos to the input list
        </label>

        <div className="flex gap-2">
          <Button
            variant="secondary"
            onClick={() => {
              if (!settings.url.trim()) return toast.error("Enter a page URL")
              if (!settings.saveDir.trim())
                return toast.error("Choose a save folder")
              setPickerOpen(true)
            }}
            disabled={running}
            className="gap-2"
          >
            <LayoutGrid className="size-4" /> Browse &amp; Select…
          </Button>
          <Button
            onClick={() => {
              if (!settings.url.trim()) return toast.error("Enter a page URL")
              if (!settings.saveDir.trim())
                return toast.error("Choose a save folder")
              onDownload()
            }}
            disabled={running}
            className="gap-2"
          >
            <Download className="size-4" />
            {running ? "Working…" : "Download Videos"}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          Browse &amp; Select opens a grid of the page's videos so you can pick
          individually. Progress and log output appear below. Requires yt-dlp.
        </p>

        <VideoPickerDialog
          open={pickerOpen}
          onOpenChange={setPickerOpen}
          url={settings.url}
          onPick={onDownloadUrls}
        />
      </CardContent>
    </Card>
  )
}
