import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

type Option = string | { value: string; label: string }

interface Props {
  label: string
  value: string
  options: Option[]
  onChange: (v: string) => void
  disabled?: boolean
}

export function SelectField({ label, value, options, onChange, disabled }: Props) {
  const norm = options.map((o) =>
    typeof o === "string" ? { value: o, label: o } : o,
  )
  return (
    // Same reasoning as NumberField: the control sits by its label, capped so a
    // long option string can't stretch the row.
    <div className="flex min-w-0 items-center gap-3">
      <Label className="min-w-0 flex-1 truncate text-sm font-normal text-muted-foreground">
        {label}
      </Label>
      <Select value={value} onValueChange={onChange} disabled={disabled}>
        <SelectTrigger className="h-7 w-56 shrink-0">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {norm.map((o) => (
            <SelectItem key={o.value} value={o.value}>
              {o.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
