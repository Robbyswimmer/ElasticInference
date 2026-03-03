#!/usr/bin/env python3
import csv
from pathlib import Path


def load_series(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(
                (
                    int(r["time_s"]),
                    int(r["spec_replicas"]),
                    int(r["ready_replicas"]),
                    int(r["available_replicas"]),
                )
            )
    if not rows:
        raise RuntimeError(f"empty series: {csv_path}")
    return rows


def draw_svg(series, title, out_svg: Path):
    W, H = 1050, 460
    ml, mr, mt, mb = 75, 25, 55, 62
    pw, ph = W - ml - mr, H - mt - mb

    xmax = max(t for t, _, _, _ in series) or 1
    ymax = max(max(s, r, a) for _, s, r, a in series)
    ymin = 1

    def sx(x):
        return ml + (x / xmax) * pw

    def sy(y):
        if ymax == ymin:
            return mt + ph / 2
        return mt + ph - (y - ymin) / (ymax - ymin) * ph

    spec_pts = " ".join(f"{sx(t):.1f},{sy(s):.1f}" for t, s, _, _ in series)
    # ready/available are effectively identical in this trace; render as one green line.
    ra_pts = " ".join(f"{sx(t):.1f},{sy(a):.1f}" for t, _, _, a in series)

    p = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    p.append('<rect width="100%" height="100%" fill="white"/>')
    p.append(f'<text x="525" y="30" text-anchor="middle" font-size="21" font-family="Arial" font-weight="700">{title}</text>')
    p.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#fcfcfd" stroke="#e7e9ef"/>')

    for y in range(ymin, ymax + 1):
        yy = sy(y)
        p.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="#eceff4"/>')
        p.append(f'<text x="{ml-10}" y="{yy+4:.1f}" text-anchor="end" font-size="11" font-family="Arial" fill="#566">{y}</text>')

    tick_count = 16
    for i in range(tick_count + 1):
        t = int(round(xmax * i / tick_count))
        xx = sx(t)
        p.append(f'<line x1="{xx:.1f}" y1="{mt+ph}" x2="{xx:.1f}" y2="{mt+ph+4}" stroke="#6b7280"/>')
        p.append(f'<text x="{xx:.1f}" y="{mt+ph+19}" text-anchor="middle" font-size="10" font-family="Arial" fill="#5b6270">{t}s</text>')

    p.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#374151"/>')
    p.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#374151"/>')

    p.append(f'<polyline points="{spec_pts}" fill="none" stroke="#c53030" stroke-width="2.8"/>')
    p.append(f'<polyline points="{ra_pts}" fill="none" stroke="#059669" stroke-width="2.4"/>')

    prev = None
    for t, s, _, _ in series:
        if prev is None or s != prev:
            p.append(f'<circle cx="{sx(t):.1f}" cy="{sy(s):.1f}" r="3.8" fill="#c53030"/>')
        prev = s

    lx, ly = ml + pw - 235, mt + 20
    p.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+22}" y2="{ly}" stroke="#c53030" stroke-width="2.8"/><text x="{lx+30}" y="{ly+4}" font-size="12" font-family="Arial">spec</text>')
    p.append(f'<line x1="{lx}" y1="{ly+18}" x2="{lx+22}" y2="{ly+18}" stroke="#059669" stroke-width="2.4"/><text x="{lx+30}" y="{ly+22}" font-size="12" font-family="Arial">ready/available</text>')

    p.append(f'<text x="{ml+pw/2:.1f}" y="{H-18}" text-anchor="middle" font-size="12" font-family="Arial">Time since trace start</text>')
    p.append(f'<text x="20" y="{mt+ph/2:.1f}" transform="rotate(-90 20,{mt+ph/2:.1f})" text-anchor="middle" font-size="12" font-family="Arial">Replicas</text>')
    p.append('</svg>')
    out_svg.write_text("\n".join(p), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parent
    decode_csv = root / "decode_trace_1-2-4-5-4-3-2-1_spec_ready_available.csv"
    decode_svg = root / "decode_trace_1-2-4-5-4-3-2-1_spec_ready_available.svg"
    prefill_csv = root / "prefill_trace_1-2-1_spec_ready_available.csv"
    prefill_svg = root / "prefill_trace_1-2-1_spec_ready_available.svg"

    draw_svg(load_series(decode_csv), "Decode Replicas Over Time", decode_svg)
    draw_svg(load_series(prefill_csv), "Prefill Replicas Over Time", prefill_svg)


if __name__ == "__main__":
    main()
