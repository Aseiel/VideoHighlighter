import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { SelectField } from "@/components/SelectField"
import { NumberField } from "@/components/NumberField"
import {
  SUBTITLE_LANGS,
  TRANSCRIPT_LANGS,
  WHISPER_MODELS,
  type HighlighterConfig,
} from "@/lib/config"

interface Props {
  cfg: HighlighterConfig
  set: <K extends keyof HighlighterConfig>(k: K, v: HighlighterConfig[K]) => void
}

export function TranscriptTab({ cfg, set }: Props) {
  // Subtitles require a transcript, and the keyword/transcript scores only
  // count when transcript runs — same gating as the Qt tab.
  const on = cfg.use_transcript
  return (
    <div className="grid min-w-0 gap-5 md:grid-cols-2 [&>*]:min-w-0">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Transcript</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={cfg.use_transcript}
              onCheckedChange={(v) => set("use_transcript", Boolean(v))}
            />
            Enable transcript processing (Whisper)
          </label>
          <SelectField
            label="Source language"
            value={cfg.transcript_source_lang}
            options={TRANSCRIPT_LANGS}
            onChange={(v) => set("transcript_source_lang", v)}
            disabled={!on}
          />
          <SelectField
            label="Whisper model"
            value={cfg.transcript_model}
            options={WHISPER_MODELS}
            onChange={(v) => set("transcript_model", v)}
            disabled={!on}
          />
          <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_14rem] items-center gap-3">
            <Label className="min-w-0 truncate text-sm font-normal text-muted-foreground">
              Search keywords
            </Label>
            <Input
              value={cfg.search_keywords}
              onChange={(e) => set("search_keywords", e.target.value)}
              placeholder="goal, score, win"
              className="h-8 w-full"
              disabled={!on}
            />
          </div>
          <div className="space-y-2.5 border-t pt-3">
            <p className="text-xs text-muted-foreground">
              These scores only count while transcript is enabled.
            </p>
            <NumberField
              label="Keyword points"
              value={cfg.keyword_points}
              onChange={(v) => set("keyword_points", v)}
            />
            <NumberField
              label="Transcript points"
              hint="(all words)"
              value={cfg.transcript_points}
              onChange={(v) => set("transcript_points", v)}
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Subtitles</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={cfg.create_subtitles}
              onCheckedChange={(v) => set("create_subtitles", Boolean(v))}
              disabled={!on}
            />
            Generate subtitles (.srt)
          </label>
          {!on && (
            <p className="text-xs text-muted-foreground">
              Enable transcript first to generate subtitles.
            </p>
          )}
          <SelectField
            label="Source language"
            value={cfg.source_lang}
            options={SUBTITLE_LANGS}
            onChange={(v) => set("source_lang", v)}
            disabled={!on || !cfg.create_subtitles}
          />
          <SelectField
            label="Target language"
            value={cfg.target_lang}
            options={SUBTITLE_LANGS}
            onChange={(v) => set("target_lang", v)}
            disabled={!on || !cfg.create_subtitles}
          />
        </CardContent>
      </Card>
    </div>
  )
}
