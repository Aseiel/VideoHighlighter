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
    <div className="grid grid-cols-[1fr_7rem] items-center gap-3">
      <Label className="text-sm font-normal text-muted-foreground">
        {label}
        {hint && <span className="ml-1 text-xs opacity-60">{hint}</span>}
      </Label>
      <Input
        type="number"
        min={min}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-8 text-right tabular-nums"
      />
    </div>
  )
}
