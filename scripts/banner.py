"""Generate the animated profile banner (assets/banner-{dark,light}.svg).

    pip install numpy opencv-python-headless
    python scripts/banner.py

Left panel: the photo in assets/source/avatar.jpg, cut out with GrabCut and
Floyd-Steinberg dithered (serpentine scan) into 1-bit dots that print in
top-to-bottom; then a subset of dots loops through the icons in SHAPES and
back. Right panel: a system-info readout. Run it by hand after
changing the photo or ROWS - it is not part of the daily workflow.
"""

from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "assets/source/avatar.jpg"
ASSETS = ROOT / "assets"
FONT = "'JetBrains Mono','SFMono-Regular',Menlo,Consolas,'Liberation Mono',monospace"

# Photo framing, in fractions of the source image: the crop around the subject
# and the GrabCut seed rectangle inside that crop.
CROP = (0.0, 0.08, 1.0, 1.0)
SEED = (0.10, 0.06, 0.999, 0.999)

W, H = 1180, 610
MAP = (40, 76, 420, 494)        # left panel x, y, w, h
INFO = (480, 76, 660, 494)      # right panel
FRAME_HEIGHT = 0.88             # keep the top 88% of the cutout (head to waist)
ZOOM = 0.92                     # portrait size relative to the map area
GAMMA = 1.1                     # >1 pushes midtones toward empty
SHARPEN = 2.5
FADE = 0.18                     # bottom fraction that fades out instead of a hard cut
GRID = 2.0                      # px between dither cells
BAND_ROWS = 4                   # dither rows revealed per animation step
PRINT_SECONDS = 3.2

# Morph loop: TRAVELLERS dots leave the portrait, visit each shape in SHAPES, come home.
TRAVELLERS = 700
LOOP_BEGIN = PRINT_SECONDS + 1.5
PORTRAIT_HOLD = 3.5
MOVE = 1.3
HOLD = 2.2
SEED_RNG = 7

ROWS = [
    ("Subject", "Sarthak Aggarwal"),
    ("Role", "AI Engineer"),
    ("Origin", "India"),
    ("Education", "KIET Group of Institutions"),
    ("Status", "Building + Learning + Shipping"),
    ("Focus", "LLMs · RAG · Agents · Inference"),
    ("Core.Lang", "Python · TypeScript · JavaScript"),
    ("Core.ML", "PyTorch · TensorFlow · scikit-learn"),
    ("Core.GenAI", "LangChain · LangGraph · LlamaIndex"),
    ("Core.Backend", "FastAPI · PostgreSQL · MongoDB"),
    ("Core.Infra", "Docker · Linux · Git"),
    ("Side.Quest", "Indie games @ DarkLab Games"),
    ("Grid.Mail", "sarthakagg567@gmail.com"),
    ("Grid.LinkedIn", "/in/sarthakaggarwal0402"),
    ("Grid.GitHub", "sarthak98765"),
    ("Grid.X", "@sarthakagg5678"),
]

THEMES = {
    "dark": {
        "bg": "#0A101F", "panel": "#0D1628", "line": "#25344C", "muted": "#8291A8",
        "text": "#DDE7F5", "portrait": "#5EEAD4", "chrome": "#38BDF8", "ok": "#10B981",
        "live": "#F87171", "pill": "#12324A",
    },
    "light": {
        "bg": "#F6F8FA", "panel": "#FFFFFF", "line": "#CBD7E1", "muted": "#64748B",
        "text": "#172033", "portrait": "#0F766E", "chrome": "#0369A1", "ok": "#059669",
        "live": "#DC2626", "pill": "#E0F2FE",
    },
}


# ------------------------------------------------------------------ portrait

def cutout():
    """Return (gray, mask) of the subject, cropped per CROP."""
    im = cv2.imread(str(SOURCE))
    h, w = im.shape[:2]
    x0, y0, x1, y1 = int(CROP[0] * w), int(CROP[1] * h), int(CROP[2] * w), int(CROP[3] * h)
    im = im[y0:y1, x0:x1]
    ch, cw = im.shape[:2]
    rect = (int(SEED[0] * cw), int(SEED[1] * ch), int((SEED[2] - SEED[0]) * cw), int((SEED[3] - SEED[1]) * ch))
    mask = np.zeros((ch, cw), np.uint8)
    cv2.grabCut(im, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 8, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    m = np.where(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6)).apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    return gray, m


def frame(gray, mask, aspect):
    """Head-and-chest crop with the given w/h aspect, centred on the head."""
    h, w = gray.shape
    fh = int(h * FRAME_HEIGHT)
    fw = min(w, int(fh * aspect))
    top_rows = mask[: h // 5] > 0
    head_x = int(np.nonzero(top_rows)[1].mean()) if top_rows.any() else w // 2
    x0 = int(np.clip(head_x - fw / 2, 0, w - fw))
    return gray[:fh, x0:x0 + fw], mask[:fh, x0:x0 + fw]


def dither(gray, mask, cols, rows, light_is_ink):
    """Floyd-Steinberg, serpentine scan. Returns list of (row, col) that are on."""
    g = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    g = np.clip(g + SHARPEN * (g - cv2.GaussianBlur(g, (0, 0), 2)), 0, 1)
    m = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_AREA) > 127
    ink = g ** GAMMA if light_is_ink else (1 - g) ** GAMMA
    ink *= np.clip((rows - np.arange(rows)) / (rows * FADE), 0, 1)[:, None]
    ink[~m] = 0

    on = []
    for r in range(rows):
        rng = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        d = 1 if r % 2 == 0 else -1
        for c in rng:
            old = ink[r, c]
            new = 1.0 if old > 0.5 else 0.0
            err = old - new
            if new:
                on.append((r, c))
            if 0 <= c + d < cols:
                ink[r, c + d] += err * 7 / 16
            if r + 1 < rows:
                if 0 <= c - d < cols:
                    ink[r + 1, c - d] += err * 3 / 16
                ink[r + 1, c] += err * 5 / 16
                if 0 <= c + d < cols:
                    ink[r + 1, c + d] += err * 1 / 16
    return on


# -------------------------------------------------------------------- morph

# Each shape returns [(mask, share)]: dots are split between parts by share so
# small-but-important parts (nodes, buttons) aren't drowned out by big ones.

def shape_code(sz):
    """</> glyph."""
    c = np.zeros((sz, sz), np.uint8)
    th = int(sz * 0.07)
    P = lambda pts: np.array([(int(x * sz), int(y * sz)) for x, y in pts], np.int32)
    cv2.polylines(c, [P([(.30, .22), (.07, .5), (.30, .78)])], False, 255, th, cv2.LINE_AA)
    cv2.polylines(c, [P([(.70, .22), (.93, .5), (.70, .78)])], False, 255, th, cv2.LINE_AA)
    cv2.line(c, (int(.59 * sz), int(.14 * sz)), (int(.41 * sz), int(.86 * sz)), 255, th, cv2.LINE_AA)
    return [(c, 1.0)]


def shape_sparkle(sz):
    """The AI sparkle: one big four-point star and a small one."""
    c = np.zeros((sz, sz), np.uint8)

    def star(cx, cy, r, th):
        pts = []
        for k in range(256):
            a = 2 * np.pi * k / 256
            # superellipse with p < 1: sharp cusps on the axes, concave sides
            rr = r / (abs(np.cos(a)) ** .55 + abs(np.sin(a)) ** .55) ** (1 / .55)
            pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
        cv2.polylines(c, [np.array(pts, np.int32)], True, 255, th, cv2.LINE_AA)

    star(.42 * sz, .57 * sz, .42 * sz, int(sz * .05))
    star(.82 * sz, .18 * sz, .17 * sz, int(sz * .04))
    return [(c, 1.0)]


def shape_controller(sz):
    """Game controller outline with a solid d-pad and buttons - for DarkLab Games."""
    body, keys = np.zeros((sz, sz), np.uint8), np.zeros((sz, sz), np.uint8)
    px = lambda x, y: (int(x * sz), int(y * sz))
    r = int(sz * 0.2)
    cv2.circle(body, px(.27, .55), r, 255, -1)
    cv2.circle(body, px(.73, .55), r, 255, -1)
    cv2.rectangle(body, px(.27, .35), px(.73, .68), 255, -1)
    body = cv2.morphologyEx(body, cv2.MORPH_GRADIENT, np.ones((int(sz * .045),) * 2, np.uint8))
    arm = int(sz * 0.03)
    cx, cy = px(.26, .53)
    cv2.rectangle(keys, (cx - 3 * arm, cy - arm), (cx + 3 * arm, cy + arm), 255, -1)
    cv2.rectangle(keys, (cx - arm, cy - 3 * arm), (cx + arm, cy + 3 * arm), 255, -1)
    for x, y in ((.74, .44), (.74, .62), (.65, .53), (.83, .53)):
        cv2.circle(keys, px(x, y), int(sz * 0.035), 255, -1)
    return [(body, 0.6), (keys, 0.4)]


SHAPES = [("CODE", shape_code), ("GEN.AI", shape_sparkle), ("DARKLAB.GAMES", shape_controller)]


def sample(parts, n, rng):
    counts = [int(n * share) for _, share in parts]
    counts[0] += n - sum(counts)
    pts = []
    for (mask, _), k in zip(parts, counts):
        ys, xs = np.nonzero(mask > 127)
        idx = rng.choice(len(xs), k, replace=len(xs) < k)
        pts.append(np.stack([xs[idx], ys[idx]], 1))
    return np.concatenate(pts).astype(np.float32) + rng.uniform(-.5, .5, (n, 2))


def assign(src, dst):
    """Greedy nearest matching: returns dst reordered so dst[i] is src[i]'s target."""
    d = ((src[:, None, :] - dst[None, :, :]) ** 2).sum(-1)
    out = np.empty_like(dst)
    used_s = np.zeros(len(src), bool)
    used_d = np.zeros(len(dst), bool)
    for flat in np.argsort(d, axis=None):
        i, j = divmod(int(flat), len(dst))
        if not used_s[i] and not used_d[j]:
            out[i] = dst[j]
            used_s[i] = used_d[j] = True
    return out


def timeline():
    """[(t, stop)] where stop is 'P' (portrait) or a shape index."""
    t, keys = 0.0, [(0.0, "P"), (PORTRAIT_HOLD, "P")]
    t = PORTRAIT_HOLD
    for i in range(len(SHAPES)):
        t += MOVE
        keys.append((t, i))
        t += HOLD
        keys.append((t, i))
    t += MOVE
    keys.append((t, "P"))
    keys.append((t + 0.8, "P"))
    return keys


def portrait(t, light_is_ink):
    mx, my, mw, mh = MAP
    box_w, box_h = (mw - 44) * ZOOM, (mh - 92) * ZOOM
    gray, mask = frame(*cutout(), box_w / box_h)
    rows = int(min(box_h, box_w * gray.shape[0] / gray.shape[1]) / GRID)
    cols = int(rows * gray.shape[1] / gray.shape[0])
    on = dither(gray, mask, cols, rows, light_is_ink)
    ox = mx + (mw - cols * GRID) / 2
    oy = my + 44 + ((mh - 92) - rows * GRID) / 2
    n_bands = rows // BAND_ROWS + 1
    delay = lambda r: 0.3 + PRINT_SECONDS * (r // BAND_ROWS) / n_bands

    # A random subset of dots travels between the portrait and each shape.
    rng = np.random.default_rng(SEED_RNG)
    pick = set(rng.choice(len(on), TRAVELLERS, replace=False).tolist())
    home = np.array([(ox + on[i][1] * GRID, oy + on[i][0] * GRID) for i in sorted(pick)], np.float32)
    home_rows = [on[i][0] for i in sorted(pick)]

    sz = int(min(cols * GRID, rows * GRID) * 0.78)
    sx, sy = ox + (cols * GRID - sz) / 2, oy + (rows * GRID - sz) / 2
    stops, prev = [], home
    for _, fn in SHAPES:
        pts = assign(prev, sample(fn(sz), TRAVELLERS, rng) + (sx, sy))
        stops.append(pts)
        prev = pts

    keys = timeline()
    loop = keys[-1][0]
    key_times = ";".join(f"{k / loop:.4f}" for k, _ in keys)
    splines = ";".join("0 0 1 1" if a == b else "0.65 0 0.35 1" for (_, a), (_, b) in zip(keys, keys[1:]))

    bands = {}
    for i, (r, c) in enumerate(on):
        if i not in pick:
            bands.setdefault(r // BAND_ROWS, []).append(f"M{ox + c * GRID:.1f} {oy + r * GRID:.1f}h0")
    out = [f'<g stroke="{t["portrait"]}" stroke-width="{GRID * 0.8:.2f}" stroke-linecap="round" fill="none">',
           f'<animate attributeName="opacity" values="1;1;0;0;1;1" '
           f'keyTimes="0;{PORTRAIT_HOLD / loop:.4f};{(PORTRAIT_HOLD + 0.5) / loop:.4f};'
           f'{keys[-2][0] / loop:.4f};{(keys[-2][0] + 0.6) / loop:.4f};1" '
           f'dur="{loop:.1f}s" begin="{LOOP_BEGIN}s" repeatCount="indefinite"/>']
    for b, segs in sorted(bands.items()):
        out.append(f'<path class="px" style="animation-delay:{delay(b * BAND_ROWS):.2f}s" d="{"".join(segs)}"/>')
    out.append("</g>")

    out.append(f'<g fill="{t["portrait"]}">')
    for k, (hx, hy) in enumerate(home):
        vals = []
        for _, stop in keys:
            if stop == "P":
                vals.append("0 0")
            else:
                dx, dy = stops[stop][k] - (hx, hy)
                vals.append(f"{dx:.1f} {dy:.1f}")
        out.append(
            f'<circle class="px" style="animation-delay:{delay(home_rows[k]):.2f}s" cx="{hx:.1f}" cy="{hy:.1f}" '
            f'r="{GRID * 0.42:.2f}"><animateTransform attributeName="transform" type="translate" '
            f'values="{";".join(vals)}" keyTimes="{key_times}" calcMode="spline" keySplines="{splines}" '
            f'dur="{loop:.1f}s" begin="{LOOP_BEGIN}s" repeatCount="indefinite"/></circle>'
        )
    out.append("</g>")

    # print head: a bright line that rides the reveal, then idles as a slow scan
    top, bot = oy - 4, oy + rows * GRID + 4
    out.append(
        f'<rect x="{mx + 14}" y="{top}" width="{mw - 28}" height="2" fill="{t["portrait"]}" opacity=".7">'
        f'<animate attributeName="y" values="{top};{bot}" begin="0.3s" dur="{PRINT_SECONDS}s" fill="freeze"/>'
        f'<animate attributeName="opacity" values=".7;0" begin="{0.3 + PRINT_SECONDS}s" dur=".4s" fill="freeze"/></rect>'
        f'<rect x="{mx + 14}" y="{top}" width="{mw - 28}" height="1" fill="{t["portrait"]}" opacity="0">'
        f'<animate attributeName="y" values="{top};{bot}" begin="{PRINT_SECONDS + 1}s" dur="6s" repeatCount="indefinite"/>'
        f'<animate attributeName="opacity" values="0;.35;.35;0" begin="{PRINT_SECONDS + 1}s" dur="6s" repeatCount="indefinite"/></rect>'
    )

    # MODE caption bottom-right of the map: switches with whatever is on screen.
    modes = ["PORTRAIT"] + [name for name, _ in SHAPES]
    cap = []
    for m_i, label in enumerate(modes):
        want = "P" if m_i == 0 else m_i - 1
        vis = ["1" if stop == want else "0" for _, stop in keys]
        cap.append(
            f'<text x="{mx + mw - 22}" y="{my + mh - 12}" text-anchor="end" fill="{t["chrome"]}" font-size="9.5" '
            f'font-weight="700" opacity="{1 if m_i == 0 else 0}">MODE ▸ {label}'
            f'<animate attributeName="opacity" values="{";".join(vis)}" keyTimes="{key_times}" calcMode="discrete" '
            f'dur="{loop:.1f}s" begin="{LOOP_BEGIN}s" repeatCount="indefinite"/></text>'
        )
    return "\n".join(out), "".join(cap), len(on), (rows, cols)


# --------------------------------------------------------------------- panes

def corners(x, y, w, h, color, k=10):
    p = [f"M{x} {y + k}V{y}H{x + k}", f"M{x + w - k} {y}H{x + w}V{y + k}",
         f"M{x} {y + h - k}V{y + h}H{x + k}", f"M{x + w - k} {y + h}H{x + w}V{y + h - k}"]
    return f'<path d="{"".join(p)}" stroke="{color}" stroke-width="1.4" fill="none"/>'


def banner(name, t):
    light_is_ink = name == "dark"
    dots, caption, n_dots, (rows, cols) = portrait(t, light_is_ink)
    mx, my, mw, mh = MAP
    ix, iy, iw, ih = INFO
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
        "<style>"
        ".px{opacity:0;animation:on .25s linear forwards}"
        ".row{opacity:0;animation:row .35s ease-out forwards}"
        "@keyframes on{to{opacity:1}}"
        "@keyframes row{from{opacity:0;transform:translateX(8px)}to{opacity:1;transform:none}}"
        ".blink{animation:blink 1.4s ease-in-out infinite}"
        "@keyframes blink{50%{opacity:.2}}"
        "@media (prefers-reduced-motion:reduce){.px,.row{animation:none;opacity:1}.blink{animation:none}}"
        "</style>",
        f'<rect x="12" y="12" width="{W - 24}" height="{H - 24}" rx="14" fill="{t["bg"]}" stroke="{t["line"]}"/>',
        f'<line x1="12" y1="56" x2="{W - 12}" y2="56" stroke="{t["line"]}"/>',
    ]
    for i, c in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
        o.append(f'<circle cx="{38 + i * 20}" cy="34" r="6" fill="{c}"/>')
    o.append(f'<text x="{W / 2}" y="39" text-anchor="middle" fill="{t["muted"]}" font-size="13">profile.sh --live</text>')

    # --- VISUAL.MAP
    o += [
        f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="6" fill="{t["panel"]}" stroke="{t["line"]}"/>',
        f'<line x1="{mx}" y1="{my + 34}" x2="{mx + mw}" y2="{my + 34}" stroke="{t["line"]}"/>',
        f'<text x="{mx + 14}" y="{my + 22}" fill="{t["chrome"]}" font-size="12.5" font-weight="700" letter-spacing="1">VISUAL.MAP</text>',
        f'<text x="{mx + mw - 14}" y="{my + 22}" text-anchor="end" fill="{t["muted"]}" font-size="10">{cols}×{rows} / 1-BIT</text>',
        corners(mx + 14, my + 46, mw - 28, mh - 80, t["muted"]),
        dots,
        f'<text x="{mx + 22}" y="{my + mh - 12}" fill="{t["muted"]}" font-size="9.5">PTS {n_dots} · FS/SERPENTINE</text>',
        caption,
    ]

    # --- SYSTEM.INFO
    o += [
        f'<rect x="{ix}" y="{iy}" width="{iw}" height="{ih}" rx="6" fill="{t["panel"]}" stroke="{t["line"]}"/>',
        f'<line x1="{ix}" y1="{iy + 34}" x2="{ix + iw}" y2="{iy + 34}" stroke="{t["line"]}"/>',
        f'<text x="{ix + 14}" y="{iy + 22}" fill="{t["chrome"]}" font-size="12.5" font-weight="700" letter-spacing="1">SYSTEM.INFO</text>',
        f'<rect x="{ix + iw - 150}" y="{iy + 8}" width="138" height="20" rx="10" fill="{t["pill"]}"/>',
        f'<text x="{ix + iw - 81}" y="{iy + 22}" text-anchor="middle" fill="{t["chrome"]}" font-size="12" font-weight="700">@sarthak98765</text>',
        f'<circle class="blink" cx="{ix + iw - 212}" cy="{iy + 18}" r="3.5" fill="{t["live"]}"/>',
        f'<text x="{ix + iw - 202}" y="{iy + 22}" fill="{t["live"]}" font-size="11" font-weight="700">LIVE</text>',
    ]
    row_h = (ih - 90) / len(ROWS)
    for i, (k, v) in enumerate(ROWS):
        y = iy + 58 + i * row_h
        kw, vw = 7.2 * len(k), 7.2 * len(v)
        o.append(
            f'<g class="row" style="animation-delay:{0.6 + 0.14 * i:.2f}s">'
            f'<text x="{ix + 16}" y="{y}" fill="{t["muted"]}" font-size="12">{escape(k)}</text>'
            f'<line x1="{ix + 24 + kw}" y1="{y - 3.5}" x2="{ix + iw - 24 - vw}" y2="{y - 3.5}" '
            f'stroke="{t["line"]}" stroke-dasharray="1.5 3.5"/>'
            f'<text x="{ix + iw - 16}" y="{y}" text-anchor="end" fill="{t["text"]}" font-size="12">{escape(v)}</text></g>'
        )
    fy = iy + ih - 14
    o += [
        f'<line x1="{ix + 14}" y1="{fy - 18}" x2="{ix + iw - 14}" y2="{fy - 18}" stroke="{t["line"]}"/>',
        f'<circle class="blink" cx="{ix + 19}" cy="{fy - 3.5}" r="3" fill="{t["ok"]}"/>',
        f'<text x="{ix + 28}" y="{fy}" fill="{t["ok"]}" font-size="10" font-weight="700">ALL SYSTEMS NOMINAL</text>',
        f'<text x="{ix + iw - 14}" y="{fy}" text-anchor="end" fill="{t["muted"]}" font-size="10">UTC+5:30 · INDIA NODE</text>',
        "</svg>",
    ]
    return "\n".join(o)


def main():
    for name, t in THEMES.items():
        svg = banner(name, t)
        (ASSETS / f"banner-{name}.svg").write_text(svg)
        print(f"banner-{name}.svg  {len(svg) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
