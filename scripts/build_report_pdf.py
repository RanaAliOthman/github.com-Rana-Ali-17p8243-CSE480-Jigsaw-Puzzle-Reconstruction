"""Typeset the Markdown reports in report/ as PDFs beside them.

    python3 -m scripts.build_report_pdf                     # every report
    python3 -m scripts.build_report_pdf report/milestone_2_report.md

The Markdown source is the single copy of each report; this script only renders
it, so the two can never disagree. Needs `markdown` and `weasyprint`:

    python3 -m pip install markdown weasyprint
"""
import re
import sys
from pathlib import Path

import markdown
from weasyprint import CSS, HTML

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'report'

STYLE = """
@page { size: A4; margin: 20mm 18mm; @bottom-center {
    content: counter(page); font-size: 9pt; color: #777; } }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 10.5pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 0.4em; }
h2 { font-size: 14pt; margin: 1.6em 0 0.5em; border-bottom: 1px solid #ccc;
     padding-bottom: 0.15em; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 1.2em 0 0.4em; break-after: avoid; }
p, li { orphans: 2; widows: 2; }
code, pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.8pt; }
pre { background: #f6f6f4; border: 1px solid #e0e0dc; border-radius: 3px;
      padding: 0.6em 0.8em; white-space: pre-wrap; break-inside: avoid; }
code { background: #f2f2f0; padding: 0.05em 0.25em; border-radius: 2px; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 0.8em 0;
        font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #d5d5d0; padding: 0.35em 0.5em; text-align: left;
         vertical-align: top; }
th { background: #f2f2ef; }
blockquote { margin: 0.8em 0; padding: 0.3em 0 0.3em 1em;
             border-left: 3px solid #ccc; color: #444; font-style: italic; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.6em 0; }
a { color: #1a4d80; text-decoration: none; word-break: break-all; }
"""


def build(source, output=None):
    """Render one Markdown report to PDF and return the output path."""
    source = Path(source)
    output = Path(output) if output else source.with_suffix('.pdf')
    if not source.exists():
        raise FileNotFoundError(source)
    text = source.read_text(encoding='utf-8')
    body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'sane_lists'])
    heading = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
    title = heading.group(1).strip() if heading else source.stem
    html = ('<!doctype html><html><head><meta charset="utf-8">'
            '<title>%s</title></head><body>%s</body></html>'
            % (markdown.util.code_escape(title), body))
    output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(source.parent)).write_pdf(
        str(output), stylesheets=[CSS(string=STYLE)])
    return output


if __name__ == '__main__':
    sources = [Path(a) for a in sys.argv[1:]] or sorted(REPORTS.glob('*_report.md'))
    if not sources:
        raise SystemExit('no report/*_report.md found')
    for source in sources:
        path = build(source)
        print('%s (%.0f KB)' % (path, path.stat().st_size / 1024), file=sys.stderr)
