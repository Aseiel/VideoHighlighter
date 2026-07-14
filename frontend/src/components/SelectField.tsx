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
    <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_14rem] items-center gap-3">
      <Label className="min-w-0 truncate text-sm font-normal text-muted-foreground">
        {label}
      </Label>
      <Select value={value} onValueChange={onChange} disabled={disabled}>
        <SelectTrigger className="h-8 w-full">
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
