"""Generate the radar charts and stats card the profile README uses.
(The banner comes from scripts/banner.py, which is run by hand.)

    python scripts/generate.py

Stdlib only. Set GITHUB_TOKEN for exact language bytes (GraphQL); without it,
falls back to the public REST API and counts repos by primary language, and
reads the contribution calendar off the public profile page.
"""

import json
import math
import os
import re
import urllib.request
from pathlib import Path
from xml.sax.saxutils import escape

USER = "sarthak98765"
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
FONT = "'JetBrains Mono','SFMono-Regular',Menlo,Consolas,'Liberation Mono',monospace"

THEMES = {
    "dark": {
        "bg": "#0d1117", "panel": "#161b22", "border": "#30363d", "text": "#e6edf3",
        "muted": "#8b949e", "accent": "#2dd4bf", "accent2": "#a78bfa", "ok": "#3fb950",
        "grid": "#30363d",
    },
    "light": {
        "bg": "#ffffff", "panel": "#f6f8fa", "border": "#d0d7de", "text": "#1f2328",
        "muted": "#656d76", "accent": "#0f766e", "accent2": "#7c3aed", "ok": "#1a7f37",
        "grid": "#d0d7de",
    },
}


# --------------------------------------------------------------------- data

# Linguist colours for the REST fallback (GraphQL returns them directly).
LANG_COLORS = {
    "Python": "#3572A5", "Jupyter Notebook": "#DA5B0B", "JavaScript": "#f1e05a",
    "TypeScript": "#3178c6", "HTML": "#e34c26", "CSS": "#663399", "Shell": "#89e051",
    "C++": "#f34b7d", "C": "#555555", "Java": "#b07219", "Dockerfile": "#384d54",
    "Go": "#00ADD8", "Rust": "#dea584", "C#": "#178600", "Makefile": "#427819",
}


def _get(url, data=None, raw=False):
    headers = {"User-Agent": USER, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode() if raw else json.load(r)


def scrape_calendar():
    """[(date, count)] from the public contributions page - works without a token."""
    page = _get(f"https://github.com/users/{USER}/contributions", raw=True)
    dates = dict(re.findall(r'data-date="([\d-]+)" id="(contribution-day-component-[\d-]+)"', page))
    dates = {cell: day for day, cell in dates.items()}
    days = {}
    for cell, text in re.findall(r'<tool-tip[^>]*for="(contribution-day-component-[\d-]+)"[^>]*>([^<]*)', page):
        m = re.match(r"\s*([\d,]+) contribution", text)
        if cell in dates:
            days[dates[cell]] = int(m.group(1).replace(",", "")) if m else 0
    return sorted(days.items())


def streaks(calendar):
    """(current, longest). Current allows today to still be empty."""
    longest = run = 0
    for _, n in calendar:
        run = run + 1 if n else 0
        longest = max(longest, run)
    counts = [n for _, n in calendar]
    if counts and counts[-1] == 0:
        counts.pop()
    current = 0
    for n in reversed(counts):
        if not n:
            break
        current += 1
    return current, longest


def fetch_stats():
    """Return {repos, stars, followers, since, contributions, streak, longest_streak, active_days,
    languages{name: (weight, colour)}, lang_basis}."""
    if os.environ.get("GITHUB_TOKEN"):
        query = """
        query($login: String!) {
          user(login: $login) {
            createdAt
            followers { totalCount }
            contributionsCollection {
              contributionCalendar {
                totalContributions
                weeks { contributionDays { date contributionCount } }
              }
            }
            repositories(first: 100, ownerAffiliations: OWNER, isFork: false, privacy: PUBLIC) {
              totalCount
              nodes {
                stargazerCount
                languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
                  edges { size node { name color } }
                }
              }
            }
          }
        }"""
        body = json.dumps({"query": query, "variables": {"login": USER}}).encode()
        u = _get("https://api.github.com/graphql", body)["data"]["user"]
        langs = {}
        for repo in u["repositories"]["nodes"]:
            for e in repo["languages"]["edges"]:
                name = e["node"]["name"]
                langs[name] = (langs.get(name, (0,))[0] + e["size"], e["node"]["color"] or "#8b949e")
        cal = u["contributionsCollection"]["contributionCalendar"]
        calendar = [(d["date"], d["contributionCount"]) for w in cal["weeks"] for d in w["contributionDays"]]
        current, longest = streaks(calendar)
        return {
            "repos": u["repositories"]["totalCount"],
            "stars": sum(r["stargazerCount"] for r in u["repositories"]["nodes"]),
            "followers": u["followers"]["totalCount"],
            "since": u["createdAt"][:4],
            "contributions": cal["totalContributions"],
            "streak": current,
            "longest_streak": longest,
            "active_days": sum(1 for _, n in calendar if n),
            "languages": langs,
            "lang_basis": "bytes",
        }

    user = _get(f"https://api.github.com/users/{USER}")
    repos = _get(f"https://api.github.com/users/{USER}/repos?per_page=100&type=owner")
    repos = [r for r in repos if not r["fork"]]
    langs = {}
    for r in repos:
        if r["language"]:
            n = r["language"]
            langs[n] = (langs.get(n, (0,))[0] + 1, LANG_COLORS.get(n, "#8b949e"))
    calendar = scrape_calendar()
    current, longest = streaks(calendar)
    return {
        "repos": len(repos),
        "stars": sum(r["stargazers_count"] for r in repos),
        "followers": user["followers"],
        "since": user["created_at"][:4],
        "contributions": sum(n for _, n in calendar),
        "streak": current,
        "longest_streak": longest,
        "active_days": sum(1 for _, n in calendar if n),
        "languages": langs,
        "lang_basis": "repos",
    }


def merged_languages(langs):
    """Notebooks are Python in practice; fold them in so they don't split the charts."""
    langs = dict(langs)
    if "Jupyter Notebook" in langs:
        w, _ = langs.pop("Jupyter Notebook")
        langs["Python"] = (langs.get("Python", (0,))[0] + w, langs.get("Python", (0, "#3572A5"))[1])
    return sorted(langs.items(), key=lambda kv: -kv[1][0])


# -------------------------------------------------------------------- radar

def radar(title, axes, t, max_value=10, note=None):
    w, h, cx, cy, r = 460, 420, 230, 228, 118
    n = len(axes)
    pt = lambda i, v: (cx + r * v * math.sin(2 * math.pi * i / n),
                       cy - r * v * math.cos(2 * math.pi * i / n))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="{FONT}">',
        "<style>.shape{transform-origin:%dpx %dpx;animation:grow 1.1s cubic-bezier(.2,.8,.2,1) both}"
        "@keyframes grow{from{transform:scale(0);opacity:0}}"
        "@media (prefers-reduced-motion:reduce){.shape{animation:none}}</style>" % (cx, cy),
        f'<rect x="1" y="1" width="{w-2}" height="{h-2}" rx="12" fill="{t["panel"]}" stroke="{t["border"]}"/>',
        f'<text x="20" y="32" fill="{t["accent"]}" font-size="14" font-weight="700">$ cat {escape(title)}</text>',
    ]
    if note:
        out.append(f'<text x="{w-20}" y="32" text-anchor="end" fill="{t["muted"]}" font-size="11">{escape(note)}</text>')
    for ring in range(1, 6):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, ring / 5) for i in range(n)))
        out.append(f'<polygon points="{pts}" fill="none" stroke="{t["grid"]}" stroke-width="1"/>')
    for i in range(n):
        x, y = pt(i, 1)
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{t["grid"]}"/>')

    vals = [min(a["value"] / max_value, 1) for a in axes]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, v) for i, v in enumerate(vals)))
    out.append(f'<g class="shape"><polygon points="{pts}" fill="{t["accent"]}" fill-opacity=".22" '
               f'stroke="{t["accent"]}" stroke-width="2" stroke-linejoin="round"/>')
    for i, v in enumerate(vals):
        x, y = pt(i, v)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{t["accent"]}"/>')
    out.append("</g>")

    for i, a in enumerate(axes):
        x, y = pt(i, 1.17)
        dx = x - cx
        anchor = "middle" if abs(dx) < 8 else ("start" if dx > 0 else "end")
        out.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" fill="{t["text"]}" '
                   f'font-size="11.5">{escape(a["label"])}</text>')
    out.append("</svg>")
    return "\n".join(out)


def language_axes(langs, k=6):
    top = [(name, w) for name, (w, _) in merged_languages(langs)[:k]]
    total = sum(v for _, v in top) or 1
    # sqrt scale so one dominant language doesn't flatten the rest to zero
    peak = math.sqrt(top[0][1] / total) if top else 1
    return [{"label": f"{name} {100 * v / total:.0f}%", "value": 10 * math.sqrt(v / total) / peak}
            for name, v in top]


# -------------------------------------------------------------------- cards

SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"
CARD_W = 560


def stats_card(s, t):
    w, h = CARD_W, 218
    cells = [
        (s["active_days"], "Active days (1y)"), (s["repos"], "Public repos"), (s["followers"], "Followers"),
        (s["contributions"], "Contributions (1y)"), (s["streak"], "Current streak"),
        (s["longest_streak"], "Longest streak"),
    ]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{SANS}">',
        "<style>.c{opacity:0;animation:in .5s ease-out forwards}"
        "@keyframes in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}"
        "@media (prefers-reduced-motion:reduce){.c{animation:none;opacity:1}}</style>",
        f'<rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="10" fill="{t["bg"]}" stroke="{t["border"]}"/>',
        f'<text x="26" y="40" fill="{t["accent"]}" font-size="19" font-weight="600">{USER}</text>',
        f'<text x="{w - 26}" y="40" text-anchor="end" fill="{t["muted"]}" font-size="13">at a glance · since {s["since"]}</text>',
        f'<line x1="26" y1="56" x2="{w - 26}" y2="56" stroke="{t["border"]}"/>',
    ]
    col_w = (w - 52) / 3
    for i, (value, label) in enumerate(cells):
        x, y = 26 + (i % 3) * col_w, 96 + (i // 3) * 66
        out.append(
            f'<g class="c" style="animation-delay:{0.08 * i:.2f}s">'
            f'<text x="{x:.0f}" y="{y}" fill="{t["text"]}" font-size="30" font-weight="700">{value}</text>'
            f'<text x="{x:.0f}" y="{y + 22}" fill="{t["muted"]}" font-size="13">{label}</text></g>'
        )
    out.append("</svg>")
    return "\n".join(out)


def human_size(n):
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000:
            return f"{n:.3g} {unit}"
        n /= 1000
    return f"{n:.3g} TB"


def languages_card(s, t, k=8):
    langs = merged_languages(s["languages"])
    total = sum(w for _, (w, _) in langs) or 1
    top = langs[:k]
    rows = (len(top) + 1) // 2
    w, h = CARD_W, 118 + rows * 26
    bar_x, bar_w, bar_y = 26, w - 52, 76
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{SANS}">',
        "<style>.seg{transform-box:fill-box;transform-origin:left;animation:grow .9s cubic-bezier(.2,.8,.2,1) both}"
        "@keyframes grow{from{transform:scaleX(0)}}"
        ".r{opacity:0;animation:in .4s ease-out forwards}@keyframes in{to{opacity:1}}"
        "@media (prefers-reduced-motion:reduce){.seg,.r{animation:none;opacity:1}}</style>",
        # code-review glyph, like GitHub's octicon
        f'<path transform="translate(26 18)" fill="{t["muted"]}" d="M1.75 1h12.5c.966 0 1.75.784 1.75 1.75v8.5A1.75 1.75 0 0 1 14.25 13H8.061l-2.574 2.573A1.458 1.458 0 0 1 3 14.543V13H1.75A1.75 1.75 0 0 1 0 11.25v-8.5C0 1.784.784 1 1.75 1ZM1.5 2.75v8.5c0 .138.112.25.25.25h2a.75.75 0 0 1 .75.75v2.19l2.72-2.72a.749.749 0 0 1 .53-.22h6.5a.25.25 0 0 0 .25-.25v-8.5a.25.25 0 0 0-.25-.25H1.75a.25.25 0 0 0-.25.25Zm5.28 1.72a.75.75 0 0 1 0 1.06L5.31 7l1.47 1.47a.751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018l-2-2a.75.75 0 0 1 0-1.06l2-2a.75.75 0 0 1 1.06 0Zm2.44 0a.75.75 0 0 1 1.06 0l2 2a.75.75 0 0 1 0 1.06l-2 2a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L10.69 7 9.22 5.53a.75.75 0 0 1 0-1.06Z"/>',
        f'<text x="52" y="32" fill="{t["accent"]}" font-size="19" font-weight="500">{len(langs)} Languages</text>',
        f'<text x="{w / 2}" y="62" text-anchor="middle" fill="{t["accent"]}" font-size="15">Most used languages</text>',
        f'<clipPath id="bar"><rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="8" rx="4"/></clipPath>',
        '<g clip-path="url(#bar)">',
    ]
    x = bar_x
    for i, (name, (wt, color)) in enumerate(langs):
        seg = bar_w * wt / total
        out.append(f'<rect class="seg" style="animation-delay:{0.05 * i:.2f}s" x="{x:.2f}" y="{bar_y}" '
                   f'width="{max(seg, 0.5) + 1:.2f}" height="8" fill="{color}"/>')
        x += seg
    out.append("</g>")
    col_w = (w - 52) / 2
    for i, (name, (wt, color)) in enumerate(top):
        cx, y = 26 + (i % 2) * col_w, 112 + (i // 2) * 26
        size = human_size(wt) if s["lang_basis"] == "bytes" else f"{wt} repo{'s' if wt != 1 else ''}"
        out.append(
            f'<g class="r" style="animation-delay:{0.4 + 0.05 * i:.2f}s">'
            f'<circle cx="{cx + 6:.0f}" cy="{y - 5}" r="5" fill="{color}"/>'
            f'<text x="{cx + 20:.0f}" y="{y}" fill="{t["text"]}" font-size="15">{escape(name)}</text>'
            f'<text x="{cx + col_w - 70:.0f}" y="{y}" text-anchor="end" fill="{t["muted"]}" font-size="13">{size}</text>'
            f'<text x="{cx + col_w - 12:.0f}" y="{y}" text-anchor="end" fill="{t["muted"]}" font-size="13">'
            f'{100 * wt / total:.2f}%</text></g>'
        )
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------- main

def main():
    ASSETS.mkdir(exist_ok=True)
    skills = json.loads((ASSETS / "skills.json").read_text())
    stats = fetch_stats()
    basis = "by code size" if stats["lang_basis"] == "bytes" else "by repo count"

    for name, t in THEMES.items():
        (ASSETS / f"radar-{name}.svg").write_text(radar(skills["title"], skills["axes"], t, note="self-rated"))
        (ASSETS / f"radar-langs-{name}.svg").write_text(
            radar("languages.radar", language_axes(stats["languages"]), t, note=basis))
        (ASSETS / f"card-stats-{name}.svg").write_text(stats_card(stats, t))
        (ASSETS / f"card-languages-{name}.svg").write_text(languages_card(stats, t))
    print(json.dumps({k: v for k, v in stats.items() if k != "languages"}))


if __name__ == "__main__":
    main()
