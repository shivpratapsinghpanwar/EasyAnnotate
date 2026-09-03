"""A standalone HTML report for a dataset.

A JSON payload is what a program wants. A person deciding whether a dataset is
ready to train wants to *see* the class imbalance, and wants to be able to send
that to a colleague. This writes one self-contained file -- no CDN, no assets
folder, no server -- that opens in any browser and survives being emailed.

Charts are hand-written SVG for the same reason: a report that depends on a
charting library downloading at open time is a report that is blank in six
months, or on a machine behind a firewall.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .. import stats
from ..annotations import formats
from ..annotations.model import AnnotationError
from ..registry import Param, operation
from ..types import Action, Context, OpResult

REPORT_PARAMS = (
    Param("folder", "dir", "Dataset to report on.", required=True, must_exist=True),
    Param("output", "file", "Where to write the .html file.", default="dataset-report.html"),
    Param(
        "source_format",
        "choice",
        "Input format; 'auto' sniffs the layout.",
        default="auto",
        choices=("auto",) + formats.FORMATS,
    ),
    Param("title", "str", "Heading shown at the top of the report.", default=""),
    Param("recursive", "bool", "Include sub-folders.", default=True),
    Param("json_too", "bool", "Also write the raw statistics as .json beside it.", default=False),
)


@operation(
    "report.html",
    title="Dataset report",
    category="report",
    summary="Write a self-contained HTML report: class balance, object sizes, findings and fixes.",
    params=REPORT_PARAMS,
    destructive=True,
    notes=(
        "One file, no dependencies, opens anywhere. Contains the same findings\n"
        "inspect.health prints, plus the distributions that are hard to read as\n"
        "numbers -- class balance and object size especially.\n"
        "\n"
        "Written for handing over: attach it to the ticket that says the dataset\n"
        "is ready, and the reviewer can see for themselves whether it is."
    ),
)
def html_report(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="report.html", dry_run=ctx.dry_run)
    folder: Path = params["folder"]
    output: Path = Path(params["output"])

    try:
        dataset = formats.read_dataset(
            folder, params["source_format"], recursive=params["recursive"], ctx=ctx
        )
    except AnnotationError as exc:
        result.fail(str(exc))
        return result

    ctx.emit(1, 2, "analysing")
    payload = stats.analyse(dataset)
    # The folder name, not the full path. A report is made to be shared, and on
    # a hosted service the absolute path leaks the server's layout and the
    # dataset's private id to anyone the file is forwarded to. The full path is
    # still in the JSON sidecar for whoever actually needs it.
    payload["source"] = folder.name or str(folder)
    ctx.emit(2, 2, "rendering")

    heading = params["title"] or folder.name
    document = _render(heading, payload)

    result.record(Action.WRITE, output, output, "html report")
    if not ctx.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(document, encoding="utf-8")
        if params["json_too"]:
            json_path = output.with_suffix(".json")
            json_path.write_text(
                json.dumps({**payload, "source_path": str(folder)}, indent=2), encoding="utf-8"
            )
            result.record(Action.WRITE, json_path, json_path, "raw statistics")

    totals = payload["totals"]
    result.counts.update(
        {
            "images": totals["images"],
            "objects": totals["objects"],
            "classes": totals["classes"],
            "findings": len(payload["findings"]),
        }
    )
    result.warnings.extend(dataset.warnings[:10])
    result.data.update(
        {
            "report": output,
            "size_bytes": len(document.encode("utf-8")),
            "findings": payload["findings"],
        }
    )
    return result


# ------------------------------------------------------------------ render

_CSS = """
:root {
  --bg: #f7f8fa; --panel: #ffffff; --ink: #1a1d23; --muted: #5c6470;
  --line: #e3e6eb; --accent: #0b6ef4; --amber: #d98200; --red: #c8352c;
  --green: #1c8a5a;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --panel: #1c1f25; --ink: #e8eaee; --muted: #9aa3b0;
    --line: #2b2f37; --accent: #5aa2ff; --amber: #f0a72a; --red: #ff6b60;
    --green: #48c98a;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.5rem; background: var(--bg); color: var(--ink);
  font: 15px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 62rem; margin: 0 auto; }
h1 { font-size: 1.7rem; margin: 0 0 .25rem; letter-spacing: -.02em; }
h2 { font-size: 1.05rem; margin: 2.5rem 0 .9rem; text-transform: uppercase;
     letter-spacing: .08em; color: var(--muted); font-weight: 600; }
.sub { color: var(--muted); margin: 0 0 2rem; font-size: .9rem; }
.cards { display: grid; gap: .75rem; grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr)); }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 1rem; }
.card .n { font-size: 1.75rem; font-weight: 650; letter-spacing: -.02em; }
.card .k { color: var(--muted); font-size: .78rem; text-transform: uppercase;
           letter-spacing: .06em; margin-top: .15rem; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
         padding: 1.25rem; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: .78rem;
     text-transform: uppercase; letter-spacing: .05em; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: 0; }
.finding { background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--line);
           border-radius: 8px; padding: .9rem 1.1rem; margin-bottom: .6rem; }
.finding.error { border-left-color: var(--red); }
.finding.warning { border-left-color: var(--amber); }
.finding.note { border-left-color: var(--accent); }
.finding .t { font-weight: 620; }
.finding .d { color: var(--muted); font-size: .9rem; margin-top: .2rem; }
.finding .f { font-size: .9rem; margin-top: .5rem; }
.finding .f b { font-weight: 600; }
.tag { display: inline-block; font-size: .68rem; text-transform: uppercase;
       letter-spacing: .07em; padding: .1rem .45rem; border-radius: 4px;
       vertical-align: .12em; margin-right: .5rem; font-weight: 700; color: #fff; }
.tag.error { background: var(--red); } .tag.warning { background: var(--amber); }
.tag.note { background: var(--accent); }
.bar { height: 1.4rem; border-radius: 3px; background: var(--accent); min-width: 2px; }
.ok { color: var(--green); font-weight: 600; }
footer { margin-top: 3rem; color: var(--muted); font-size: .8rem;
         border-top: 1px solid var(--line); padding-top: 1rem; }
code { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: .85em;
       background: rgba(128,128,128,.14); padding: .1rem .35rem; border-radius: 4px; }
"""


def _render(heading: str, payload: dict[str, Any]) -> str:
    totals = payload["totals"]
    parts: list[str] = []
    add = parts.append

    add("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    add("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    add(f"<title>{_e(heading)} — dataset report</title>")
    add(f"<style>{_CSS}</style></head><body><div class='wrap'>")

    add(f"<h1>{_e(heading)}</h1>")
    add(
        f"<p class='sub'>{_e(str(payload.get('source', '')))} &middot; "
        f"detected format <code>{_e(payload.get('source_format', '?'))}</code></p>"
    )

    add("<div class='cards'>")
    for key, label in (
        ("images", "images"),
        ("objects", "objects"),
        ("classes", "classes"),
        ("backgrounds", "backgrounds"),
        ("polygon_objects", "polygons"),
    ):
        add(f"<div class='card'><div class='n'>{totals.get(key, 0):,}</div>"
            f"<div class='k'>{label}</div></div>")
    add("</div>")

    # -- findings
    findings = payload["findings"]
    add("<h2>Findings</h2>")
    if not findings:
        add("<div class='panel'><span class='ok'>Nothing to flag.</span> "
            "Class balance, object sizes and geometry all look reasonable.</div>")
    for finding in findings:
        severity = _e(finding["severity"])
        add(
            f"<div class='finding {severity}'>"
            f"<div class='t'><span class='tag {severity}'>{severity}</span>"
            f"{_e(finding['issue'])}</div>"
            f"<div class='d'>{_e(finding['detail'])}</div>"
            f"<div class='f'><b>Fix:</b> {_e(finding['fix'])}</div></div>"
        )

    # -- class balance
    classes = payload["per_class"]
    add("<h2>Class balance</h2><div class='panel'>")
    if classes:
        balance = payload["balance"]
        add(
            f"<p class='sub' style='margin:0 0 1rem'>Commonest class outnumbers the "
            f"rarest by <b>{balance['ratio']:g}&times;</b> "
            f"(imbalance index {balance['gini']:g}, 0 = even).</p>"
        )
        peak = max(c["objects"] for c in classes)
        add("<table><tr><th>Class</th><th class='num'>Objects</th>"
            "<th class='num'>Images</th><th class='num'>Share</th>"
            "<th style='width:38%'>Distribution</th></tr>")
        for entry in classes:
            width = max(1, round(100 * entry["objects"] / peak))
            add(
                f"<tr><td>{_e(entry['class'])}</td>"
                f"<td class='num'>{entry['objects']:,}</td>"
                f"<td class='num'>{entry['images']:,}</td>"
                f"<td class='num'>{entry['share']:.1%}</td>"
                f"<td><div class='bar' style='width:{width}%'></div></td></tr>"
            )
        add("</table>")
    else:
        add("<p>No classes found.</p>")
    add("</div>")

    # -- object sizes
    sizes = payload["object_sizes"]
    total_objects = max(1, sum(sizes.values()))
    add("<h2>Object size</h2><div class='panel'>")
    add("<table><tr><th>Bucket</th><th class='num'>Objects</th><th class='num'>Share</th>"
        "<th style='width:38%'></th></tr>")
    explanations = {
        "tiny": f"under {stats.TINY_FRACTION:.2%} of the frame",
        "small": f"under {stats.SMALL_AREA:,} px&sup2; (COCO 'small')",
        "medium": f"under {stats.MEDIUM_AREA:,} px&sup2;",
        "large": "everything bigger",
    }
    for bucket in ("tiny", "small", "medium", "large"):
        count = sizes.get(bucket, 0)
        share = count / total_objects
        add(
            f"<tr><td><b>{bucket}</b><br><span class='k' style='font-size:.78rem;color:var(--muted)'>"
            f"{explanations[bucket]}</span></td>"
            f"<td class='num'>{count:,}</td><td class='num'>{share:.1%}</td>"
            f"<td><div class='bar' style='width:{max(1, round(100 * share))}%'></div></td></tr>"
        )
    add("</table></div>")

    # -- shape of the data
    area = payload["relative_area"]
    ratio = payload["aspect_ratio"]
    per_image = payload["objects_per_image"]
    add("<h2>Distributions</h2><div class='panel'><table>")
    add("<tr><th>Measure</th><th class='num'>Min</th><th class='num'>p25</th>"
        "<th class='num'>Median</th><th class='num'>p75</th><th class='num'>Max</th></tr>")
    add(
        "<tr><td>Object area, as a fraction of its image</td>"
        + "".join(
            f"<td class='num'>{area[k]:.4f}</td>" for k in ("min", "p25", "median", "p75", "max")
        )
        + "</tr>"
    )
    add(
        "<tr><td>Object aspect ratio (width / height)</td>"
        + "".join(
            f"<td class='num'>{ratio[k]:.2f}</td>" for k in ("min", "p25", "median", "p75", "max")
        )
        + "</tr>"
    )
    add("</table>")
    add(
        f"<p class='sub' style='margin:1rem 0 0'>Objects per image: mean "
        f"<b>{per_image['mean']}</b>, max <b>{per_image['max']}</b>.</p>"
    )
    add("</div>")

    # -- composition
    add("<h2>Composition</h2><div class='panel'><table>")
    add("<tr><th>Split</th><th class='num'>Images</th></tr>")
    for name, count in payload["splits"].items():
        add(f"<tr><td>{_e(name)}</td><td class='num'>{count:,}</td></tr>")
    add("</table>")
    image_sizes = payload["image_sizes"]
    add(f"<p class='sub' style='margin:1rem 0 .4rem'>{image_sizes['distinct']} distinct "
        f"image resolution(s). Most common:</p><table>")
    add("<tr><th>Resolution</th><th class='num'>Images</th></tr>")
    for entry in image_sizes["most_common"]:
        add(f"<tr><td><code>{_e(entry['size'])}</code></td>"
            f"<td class='num'>{entry['images']:,}</td></tr>")
    add("</table></div>")

    add(
        "<footer>Generated by Data Factory <code>report.html</code>. "
        "Thresholds for 'tiny', 'imbalanced' and 'too few examples' are documented "
        "in <code>data_factory/core/stats.py</code>.</footer>"
    )
    add("</div></body></html>")
    return "".join(parts)


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)
