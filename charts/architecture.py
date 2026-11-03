"""Draws the architecture figures in the README: `mmomez` to architecture.svg and .png, `magree` to architecture-magree.svg and .png. python3 charts/architecture.py
Needs rsvg-convert for the PNG. The wiring follows three(), MoME and Agree in methods.py and Block and LM in env.py."""
import os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); out = []
FILL = dict(block="#FFE6CC", tab="#E1D5E7", op="#D5E8D4", lin="#DAE8FC", new="#F8C9B8", voc="#FFF2CC", inp="#FFE6CC")
EDGE = dict(block="#7A5A36", tab="#5B3F73", op="#3F6B3D", lin="#4A6A9A", new="#C2452D", voc="#8A7420", inp="#7A1F1F")


def text(x, y, s, size=15, anchor="middle", weight="normal", style="normal", fill="#111", family="Helvetica, Arial, sans-serif"):
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    out.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" font-style="{style}" fill="{fill}" text-anchor="{anchor}" dominant-baseline="central">{s}</text>')


def box(cx, cy, w, h, kind, s, sub=None, r=7, stack=0, size=15):
    for i in range(stack, 0, -1):
        out.append(f'<rect x="{cx - w / 2 + 7 * i}" y="{cy - h / 2 - 7 * i}" width="{w}" height="{h}" rx="{r}" fill="{FILL[kind]}" fill-opacity="{.75 - .22 * i}" stroke="{EDGE[kind]}" stroke-opacity="{.75 - .22 * i}" stroke-width="1.2"/>')
    sw = 2.2 if kind == "new" else 1.4
    out.append(f'<rect x="{cx - w / 2}" y="{cy - h / 2}" width="{w}" height="{h}" rx="{r}" fill="{FILL[kind]}" stroke="{EDGE[kind]}" stroke-width="{sw}"/>')
    if sub:
        text(cx, cy - 9, s, size); text(cx, cy + 10, sub, size - 2.5, fill="#333")
    else:
        text(cx, cy, s, size)


def trap(cx, cy, w, s, kind="lin", h=34):
    """A linear layer, drawn like the trapezoids in the Engram paper's figure."""
    sw = 2.2 if kind == "new" else 1.4; i = w * .24
    out.append(f'<polygon points="{cx - w / 2},{cy + h / 2} {cx + w / 2},{cy + h / 2} {cx + w / 2 - i},{cy - h / 2} {cx - w / 2 + i},{cy - h / 2}" fill="{FILL[kind]}" stroke="{EDGE[kind]}" stroke-width="{sw}" stroke-linejoin="round"/>')
    text(cx, cy + 1, s, 14)


def line(*p, arrow=True, dash=False, width=1.5, color="#111"):
    d = " ".join(f"{x},{y}" for x, y in p); m = ' marker-end="url(#a)"' if arrow else ""; da = ' stroke-dasharray="5 4"' if dash else ""
    out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{m}{da}/>')


def op(cx, cy, kind):
    out.append(f'<circle cx="{cx}" cy="{cy}" r="10" fill="{"#fff" if kind == "+" else FILL["op"]}" stroke="#111" stroke-width="1.4"/>')
    if kind == "+":
        out.append(f'<path d="M{cx - 10},{cy}H{cx + 10}M{cx},{cy - 10}V{cy + 10}" stroke="#111" stroke-width="1.3"/>')
    else:
        q = 7.07; out.append(f'<path d="M{cx - q},{cy - q}L{cx + q},{cy + q}M{cx + q},{cy - q}L{cx - q},{cy + q}" stroke="#111" stroke-width="1.3"/>')


def dot(x, y):
    out.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#111"/>')


def chips(words, cx, cy, gap=5, dashed=False):
    wid = [len(w) * 7.6 + 15 for w in words]; x = cx - (sum(wid) + gap * (len(words) - 1)) / 2; left = x
    for w, n in zip(wid, words):
        out.append(f'<rect x="{x}" y="{cy - 14}" width="{w}" height="28" rx="6" fill="{FILL["tab"]}" stroke="{EDGE["tab"]}" stroke-width="1.2"/>'); text(x + w / 2, cy, n, 14.5); x += w + gap
    if dashed: out.append(f'<rect x="{left - 7}" y="{cy - 20}" width="{x - gap - left + 14}" height="40" fill="none" stroke="#555" stroke-width="1.2" stroke-dasharray="5 4"/>')


W, H = 1240, 920
X = 225; P = (500, 40, 730, 800); B = 790  # whole-model column, zoom panel, hidden state bus


def model(table, zero):
    """The whole model, bottom to top, and the panel that zooms into its last block."""
    chips(["The", "capital", "of", "France", "is"], X, 872)
    line((X, 856), (X, 826)); box(X, 806, 180, 38, "voc", "Vocab Embedding")
    line((X, 787), (X, 762)); box(X, 742, 200, 38, "block", "Block 1 with MoE")
    line((X, 723), (X, 695)); box(X, 668, 180, 50, "tab", "Word Table", "key: one token"); op(X, 620, "+")
    line((X, 643), (X, 632)); line((X, 702), (X - 118, 702), (X - 118, 620), (X - 12, 620))
    line((X, 610), (X, 590)); box(X, 570, 200, 38, "block", "Blocks 2 to N-1 with MoE", stack=2, size=14.5)

    out.append(f'<rect x="{X - 118}" y="210" width="236" height="318" rx="20" fill="#F5F5F5" stroke="#111" stroke-width="1.5"/>')
    text(X + 58, 508, "Last block", 14, fill="#333")
    line((X, 551), (X, 485), arrow=False); line((X, 490), (X, 470)); box(X, 450, 170, 40, "block", table)
    line((X, 430), (X, 402)); op(X, 390, "+"); line((X, 490), (X - 95, 490), (X - 95, 390), (X - 12, 390))
    line((X, 380), (X, 350)); box(X, 330, 150, 38, "block", "MoE"); line((X, 311), (X, 267)); op(X, 255, "+"); line((X, 368), (X - 95, 368), (X - 95, 255), (X - 12, 255))

    line((X, 245), (X, 186)); box(X, 158, 180, 50, "tab", "N-gram Table", "key: last 2 and 3 tokens", size=15); op(X, 112, "+")
    if zero:
        line((X + 90, 158), (X + 108, 158), arrow=False); box(X + 155, 158, 94, 30, "new", "starts at zero", size=13)
    line((X, 133), (X, 124)); line((X, 200), (X - 118, 200), (X - 118, 112), (X - 12, 112))
    line((X, 102), (X, 86)); box(X, 67, 180, 36, "voc", "Norm + Tied Output"); line((X, 49), (X, 18)); text(X + 10, 26, "next-word scores", 14, "start", fill="#333")

    line((X + 85, 430), (P[0], P[1]), arrow=False, dash=True, width=1.2); line((X + 85, 470), (P[0], P[1] + P[3]), arrow=False, dash=True, width=1.2)
    out.append(f'<rect x="{P[0]}" y="{P[1]}" width="{P[2]}" height="{P[3]}" fill="#F5F5F5" stroke="#111" stroke-width="1.5"/>')
    box(575, 872, 130, 34, "inp", "Hidden State h"); line((575, 855), (575, B), arrow=False)


def zoom_mmomez():
    line((575, B), (1160, B), arrow=False)
    for x, n in ((575, "Q"), (645, "K"), (715, "V")):
        line((x, B), (x, 730)); trap(x, 713, 58, n); dot(x, B) if x != 575 else None
    A = 645; line((575, 696), (575, 230)); line((645, 696), (645, 230)); line((715, 696), (715, 312)); op(715, 300, "+"); line((715, 290), (715, 230))
    box(A, 210, 200, 40, "op", "Scaled Dot Product", size=15); line((A, 190), (A, 107)); trap(A, 90, 90, "Linear"); line((A, 73), (A, 22))

    # The row of the current token, picked by the head
    T = 850; chips(["France"], T, 748, dashed=True); line((T, 728), (T, 666))
    box(T, 640, 170, 52, "tab", "Token Table", "8 slots per token", stack=2, size=15)
    C = 1060; line((C, B), (C, 754)); dot(C, B); trap(C, 737, 84, "Router"); line((C, 720), (C, 688)); box(C, 670, 150, 36, "op", "Top 2 of 8 Slots", size=14)
    K = 955; line((T, 614), (T, 590), (K - 40, 590), (K - 40, 572)); line((C, 652), (C, 590), (K + 40, 590), (K + 40, 572))
    box(K, 552, 200, 40, "op", "Weighted Sum, per Head", size=14.5); line((K, 532), (K, 412)); text(K - 12, 470, "m", 15, "end", style="italic", family="Georgia, serif"); op(K, 400, "x")

    # The gate, then back into the values
    GL = 1160; line((GL, B), (GL, 487)); dot(GL, B); trap(GL, 470, 84, "Linear"); line((GL, 453), (GL, 419)); box(1100, 400, 150, 38, "op", "2 x Sigmoid", size=15); line((1025, 400), (K + 12, 400))
    line((K, 390), (K, 300), (727, 300))

    fx = 790
    for yy, name, rest in ((122, "m", "= w₁ slot₁ + w₂ slot₂"), (156, "gate", "= 2 σ( Linear (h) )"), (190, "V′", "= V + gate · m")):
        text(fx, yy, name, 16, "start", style="italic", family="Georgia, serif"); text(fx + 58, yy, rest, 15, "start", family="Georgia, serif")
    text(1216, 72, "Attention with MoME, last block only", 16, "end", "bold")


def zoom_magree():
    line((575, B), (1184, B), arrow=False)
    for x, n in ((575, "Q"), (645, "K"), (715, "V")):
        line((x, B), (x, 730)); trap(x, 713, 58, n); dot(x, B) if x != 575 else None
    A = 645; line((575, 696), (575, 660)); line((645, 696), (645, 660)); line((715, 696), (715, 660)); box(A, 640, 200, 40, "op", "Scaled Dot Product", size=15)
    line((A, 620), (A, 142)); op(A, 130, "+"); line((A, 120), (A, 106)); trap(A, 90, 90, "Linear"); line((A, 73), (A, 22)); text(A - 11, 400, "y", 15, style="italic", family="Georgia, serif")

    # Which tokens the head read
    T = 872
    chips(["France", "capital", "of", "The"], T, 748, dashed=True)
    line((T, 728), (T, 688)); box(T, 670, 170, 36, "op", "Top 4 Attended", size=15)
    line((745, 640), (T - 58, 640), (T - 58, 651)); text(768, 626, "weights", 13, "start", fill="#333")

    # The reader's code
    C = 1045; line((C, B), (C, 754)); dot(C, B); trap(C, 737, 84, "Router"); line((C, 720), (C, 688)); box(C, 670, 150, 36, "op", "Top 2 of 16 Codes", size=14)

    # Key, table, weighted row
    K = 955; line((T, 652), (T, 622), (K - 28, 622), (K - 28, 606)); line((C, 652), (C, 622), (K + 28, 622), (K + 28, 606))
    box(K, 588, 200, 36, "new", "Hash (Token, Code)", stack=2, size=15); line((K, 570), (K, 548))
    box(K, 522, 210, 52, "tab", "Token + Code Table", "4 tokens x 2 codes, per head", stack=2, size=15); line((K, 496), (K, 472))
    box(K, 454, 170, 36, "op", "Weighted Sum"); text(K - 12, 412, "m", 15, "end", style="italic", family="Georgia, serif")

    # The fit gate
    line((K, 436), (K, 237)); op(K, 225, "x"); dot(K, 400)
    line((K, 400), (1040, 400), (1040, 368)); EX = 1124; line((EX, B), (EX, 480)); dot(EX, B); trap(EX, 462, 92, "Linear 1", kind="new"); line((EX, 445), (EX, 368))
    box(1072, 349, 124, 38, "new", "4 x Cosine", size=15)
    GL = 1184; line((GL, B), (GL, 368)); trap(GL, 350, 84, "Linear 2")
    line((1072, 330), (1072, 295), (1128, 295)); op(1140, 295, "+"); line((GL, 333), (GL, 295), (1152, 295)); line((1140, 285), (1140, 245))
    box(1100, 225, 150, 38, "op", "2 x Sigmoid", size=15); line((1025, 225), (K + 12, 225))
    line((K, 215), (K, 130), (A + 12, 130))

    fx = 672
    for yy, name, rest in ((360, "fit", "= 4 cos( Linear 1 (h), m )"), (394, "gate", "= 2 σ( Linear 2 (h) + fit )"), (428, "output", "= y + gate · m")):
        text(fx, yy, name, 16, "start", style="italic", family="Georgia, serif"); text(fx + 58, yy, rest, 15, "start", family="Georgia, serif")
    text(1216, 72, "Attention with a gated table, last block only", 16, "end", "bold")


def draw(name, table, zoom, zero):
    out.clear()
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W * 2}" height="{H * 2}"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7.5" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8 Z" fill="#111"/></marker></defs><rect width="{W}" height="{H}" fill="#fff"/>')
    model(table, zero); zoom()
    x = 660; y = 872
    for k, n, w in (("block", "Transformer", 112), ("tab", "Table", 72), ("op", "Operation", 100), ("lin", "Linear", 78), ("new", "Changed in this repo" if zero else "New in this repo", 0)):
        out.append(f'<rect x="{x}" y="{y - 11}" width="30" height="22" rx="5" fill="{FILL[k]}" stroke="{EDGE[k]}" stroke-width="{2.2 if k == "new" else 1.4}"/>'); text(x + 37, y, n, 14, "start"); x += w + 12
    out.append("</svg>")
    svg = os.path.join(HERE, name + ".svg"); open(svg, "w").write("\n".join(out))
    subprocess.run(["rsvg-convert", "-w", "2480", svg, "-o", os.path.join(HERE, name + ".png")], check=True)


draw("architecture", "Attention + MoME", zoom_mmomez, True)
draw("architecture-magree", "Attention + Table", zoom_magree, False)
