import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface Props {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  step?: number
  hint?: string
}

export function NumberField({ label, value, onChange, min = 0, step = 1, hint }: Props) {
  return (
    // min-w-0 on the grid and the label: grid children default to
    // min-width:auto, which makes long labels force the row wider than the card
    // instead of truncating — that overflows the whole layout.
    <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_5rem] items-center gap-3">
      <Label className="min-w-0 truncate text-sm font-normal text-muted-foreground">
        {label}
        {hint && <span className="ml-1 text-xs opacity-60">{hint}</span>}
      </Label>
      <Input
        type="number"
        min={min}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-8 w-full text-right tabular-nums"
      />
    </div>
  )
}
