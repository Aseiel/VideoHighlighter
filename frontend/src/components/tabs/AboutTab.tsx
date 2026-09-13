import { useEffect, useState } from "react"
import { Film } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { getAbout, type AboutInfo } from "@/lib/api"

const Link = ({ href, children }: { href: string; children: React.ReactNode }) => (
  <a
    href={href}
    target="_blank"
    rel="noreferrer"
    className="text-primary hover:underline"
  >
    {children}
  </a>
)

export function AboutTab() {
  const [info, setInfo] = useState<AboutInfo | null>(null)

  useEffect(() => {
    void getAbout().then((r) => r.ok && setInfo(r))
  }, [])

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3">
        <div className="grid size-12 place-items-center rounded-xl bg-primary/15 text-primary">
          <Film className="size-6" />
        </div>
        <div>
          <h2 className="text-lg font-semibold">
            Video Highlighter {info?.edition && `(${info.edition})`}
          </h2>
          <p className="text-sm text-muted-foreground">
            {info ? `Version ${info.version} — free & open source (AGPLv3)` : "…"}
          </p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">VideoHighlighter Pro</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <p>
            You&apos;re running the free, open-source edition — face identity,
            expressions, the report and the assistant are all here.{" "}
            <strong>Pro</strong> teaches the app a vocabulary of its own:
            categories from your own example frames, search by example,
            open-vocabulary detection, a live overlay, and a commercial licence.
          </p>
          <p>
            {info ? (
              <Link href={info.website}>Learn more / Get Pro</Link>
            ) : (
              <Link href="https://aseiel.github.io/VideoHighlighter-site/">
                Learn more / Get Pro
              </Link>
            )}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Contact &amp; Support</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          {info && (
            <>
              <p>
                Email:{" "}
                <Link
                  href={`mailto:${info.support_email}?subject=VideoHighlighter%20support`}
                >
                  {info.support_email}
                </Link>
              </p>
              <p>
                Discord: <Link href={info.discord}>Join the server</Link>
              </p>
              <p>
                Website: <Link href={info.website}>{info.website}</Link>
              </p>
              <p>
                Source code: <Link href={info.repo}>{info.repo}</Link>
              </p>
              {info.log_path && (
                <p className="pt-1 text-xs text-muted-foreground">
                  Reporting a bug? Attach the debug log: {info.log_path}
                </p>
              )}
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Legal</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>© 2026 Przemysław Kreft and contributors</p>
          <p>
            Licensed under the{" "}
            <Link href="https://www.gnu.org/licenses/agpl-3.0.html">AGPLv3</Link>.
          </p>
          <p className="text-xs">
            Third-party components include PySide6 (Qt) and FFmpeg, under their
            respective licences.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
