import time
import tkinter as tk
import numpy as np

# Grid and Font dimensions
COLS, ROWS = 135, 52
CW, CH = 7, 13
WIDTH, HEIGHT = COLS * CW, ROWS * CH

# Crisp 5x9 Bitmap Font (embedded so zero external libraries are needed)
FONT_HEX = {
    "A": "20508888f88888",
    "B": "f08888f08888f0",
    "C": "70888080808870",
    "D": "e09088888890e0",
    "E": "f88080f08080f8",
    "F": "f88080f0808080",
    "G": "708880b8888870",
    "H": "888888f8888888",
    "I": "70202020202070",
    "J": "38101010909060",
    "K": "8890a0c0a09088",
    "L": "808080808080f8",
    "M": "88d8a888888888",
    "N": "88c8a898888888",
    "O": "70888888888870",
    "P": "f08888f0808080",
    "Q": "70888888a89068",
    "R": "f08888f0a09088",
    "S": "70888070088870",
    "T": "f8202020202020",
    "U": "88888888888870",
    "V": "88888888885020",
    "W": "88888888a8d888",
    "X": "88885020508888",
    "Y": "88885020202020",
    "Z": "f80810204080f8",
    "0": "708898a8c88870",
    "1": "20602020202070",
    "2": "708808304080f8",
    "3": "70880830088870",
    "4": "10305090f81010",
    "5": "f880f008088870",
    "6": "708880f0888870",
    "7": "f8081020404040",
    "8": "70888870888870",
    "9": "70888878088870",
    "_": "000000000000f8",
    ".": "00000000006060",
    ":": "00606000606000",
    "+": "002020f8202000",
    "-": "000000f8000000",
    "|": "20202020202020",
    "=": "0000f800f80000",
    "*": "008850f8508800",
    "#": "50f85050f85000",
    "(": "10204040402010",
    ")": "40201010102040",
    "[": "70404040404070",
    "]": "70101010101070",
    ",": "00000000002040",
    "~": "0076dc00000000",
    "/": "08102020408000",
    ">": "80402010204080",
    "<": "10204080402010",
    "&": "60906050909068",
    " ": "00000000000000",
}


def build_font_lut():
    """Builds a fast numpy lookup table for 128 ASCII characters"""
    lut = np.zeros((128, CH, CW), dtype=np.float32)
    for ch, hx in FONT_HEX.items():
        mat = np.zeros((CH, CW), dtype=np.float32)
        for r in range(7):
            byte = int(hx[r * 2 : r * 2 + 2], 16)
            for c in range(5):
                if byte & (1 << (7 - c)):
                    mat[r + 3, c + 1] = 1.0
        lut[ord(ch)] = mat
        if "A" <= ch <= "Z":
            lut[ord(ch.lower())] = mat
    return lut


def ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fbm(x, y):
    v = 0.500 * (np.sin(x * 1.7 + np.cos(y * 1.3)) * 0.5 + 0.5)
    v += 0.250 * (np.sin(x * 3.5 - np.sin(y * 2.9)) * 0.5 + 0.5)
    v += 0.125 * (np.cos(x * 7.1 + y * 5.3) * 0.5 + 0.5)
    return v / 0.875


# Precompute static textures once so every frame calculates instantly
U_GRID, V_GRID = np.meshgrid(
    np.linspace(0, 1, COLS, dtype=np.float32),
    np.linspace(0, 1, ROWS, dtype=np.float32),
)
DU = U_GRID - 0.5
DV = V_GRID - 0.5
TX_STATIC = fbm(U_GRID * 120 + 100, V_GRID * 120 + 100)
SKIN_BASE_STATIC = (
    np.array([0.82, 0.48, 0.29], np.float32) * (0.90 + 0.20 * TX_STATIC)[..., None]
)
HAIR_STATIC = fbm(U_GRID * 90 + V_GRID * 30, V_GRID * 200) * 0.4 + 0.7
BY_STATIC = V_GRID - (0.09 + 0.42 * (U_GRID - 0.48) ** 2 - 0.05 * DU)
EB_STATIC = ss(0.42, 0.05, np.abs(DU)) * np.exp(
    -((BY_STATIC / (0.048 + 0.012 * np.abs(DU))) ** 2)
)
OU_STATIC = ss(0.0, 0.30, np.abs(DV))
CR_STATIC = np.exp(-(((V_GRID - 0.20) / 0.09) ** 2))


def render_eye(oa, gx, gy, rp, sp):
    r_pos = 0.22 * oa * np.clip(1.0 - (DU / 0.42) ** 2, 0.0, 1.0)
    r_neg = -0.17 * oa * np.clip(1.0 - (DU / 0.42) ** 2, 0.0, 1.0)
    upper = DV < 0.0
    d_lid = np.where(upper, r_pos + DV, -r_neg - DV)
    inside = (DV > -r_pos) & (DV < -r_neg)

    lid_tone = (
        0.55 * (1 - OU_STATIC)
        + 0.85 * OU_STATIC
        - 0.30 * CR_STATIC * upper
        + 0.15 * np.exp(-(((np.abs(DV) - r_pos) / 0.028) ** 2)) * upper
        - 0.25 * np.exp(-(((np.abs(DV) - np.abs(r_neg)) / 0.04) ** 2)) * (~upper)
    )

    sh = ss(0.0, 0.05, np.abs(d_lid))
    skin = (
        SKIN_BASE_STATIC
        * np.clip(lid_tone, 0.2, 1.0)[..., None]
        * (0.4 + 0.6 * sh)[..., None]
        + np.array([0.08, 0.03, 0.01], np.float32)
        * sh[..., None]
        * (lid_tone > 0.55)[..., None]
    )

    skin = skin * (
        1 - np.clip(EB_STATIC * HAIR_STATIC * 0.85, 0, 0.92)[..., None]
    ) + np.array([0.07, 0.04, 0.02], np.float32) * np.clip(
        EB_STATIC * HAIR_STATIC, 0, 0.92
    )[..., None]

    ul = (
        np.exp(-((np.abs(DV + r_pos) / 0.022) ** 2))
        * upper
        * (0.4 + 0.6 * (np.sin(U_GRID * 260) * 0.5 + 0.5))
        * ss(0.40, 0.10, np.abs(DU))
    )
    ll = (
        np.exp(-((np.abs(DV - r_neg) / 0.015) ** 2))
        * (~upper)
        * (0.3 + 0.4 * (np.sin(U_GRID * 200) * 0.5 + 0.5))
        * ss(0.36, 0.10, np.abs(DU))
    )
    skin = skin * (1.0 - np.clip(ul + ll, 0, 0.9)[..., None] * 0.9)

    crease = (
        np.exp(-(((DV + r_pos + 0.065) / 0.016) ** 2))
        * upper
        * ss(0.38, 0.08, np.abs(DU))
        * 0.55
    )
    skin = skin * (1 - crease[..., None])

    tear = np.exp(-(((U_GRID - 0.15) ** 2 + (V_GRID - 0.505) ** 2) / 0.0012)) * oa
    skin = (
        skin * (1 - tear[..., None] * 0.65)
        + np.array([0.75, 0.35, 0.30], np.float32) * (tear * 0.65)[..., None]
    )

    wat = (
        np.exp(-(((DV - r_neg - 0.008) / 0.009) ** 2))
        * (~upper)
        * ss(0.36, 0.06, np.abs(DU))
        * oa
    )
    skin = (
        skin * (1 - wat[..., None] * 0.5)
        + np.array([0.85, 0.55, 0.48], np.float32) * (wat * 0.5)[..., None]
    )

    # Eyeball
    ex = (DU - gx) * 1.65
    ey = DV - gy
    er = np.sqrt(ex * ex + ey * ey)
    ang = np.arctan2(ey, ex)

    sc_sh = np.clip(1.0 - 0.85 * (er / 0.55) ** 2, 0.18, 1.0)
    sclera = np.array([0.85, 0.80, 0.76], np.float32) * sc_sh[..., None]

    r_iris = 0.255
    imask = ss(r_iris, r_iris - 0.018, er)
    fib = (
        np.sin(ang * 42.0 + np.sin(er * 35.0) * 1.8) * 0.5
        + 0.5
        + np.cos(ang * 84.0) * 0.25
    )
    iring = ss(r_iris, rp, er)
    icol = (
        np.array([0.76, 0.42, 0.16], np.float32) * (0.50 + 0.65 * fib)[..., None]
        + np.array([0.92, 0.62, 0.24], np.float32) * (iring * 0.35)[..., None]
    )
    limbal = ss(r_iris - 0.04, r_iris, er)
    icol *= (1.0 - 0.78 * limbal)[..., None]

    pmask = ss(rp, rp - 0.012, er)
    spec = (
        np.exp(-(((ex - 0.055) ** 2 + (ey - 0.055) ** 2) / 0.0010))
        + 0.45 * np.exp(-(((ex + 0.065) ** 2 + (ey - 0.03) ** 2) / 0.0005))
    ) * (1.0 - sp)

    eyeball = sclera * (1.0 - imask)[..., None] + icol * imask[..., None]
    eyeball = eyeball * (1.0 - pmask)[..., None]
    eyeball += np.array([1.0, 0.98, 0.95], np.float32) * np.clip(spec, 0, 1)[..., None]

    lid_sh = ss(0.0, 0.05, np.abs(DV + r_pos)) * ss(0.0, 0.03, np.abs(DV - r_neg))
    eyeball *= (0.30 + 0.70 * lid_sh)[..., None]

    if sp > 0.001:
        pr = np.sqrt((DU * 1.85) ** 2 + DV**2)
        p_rad = 0.23 * ss(0.0, 0.6, sp)
        p_mask = ss(p_rad, p_rad - 0.025, pr)

        bands = 0.55 + 0.45 * np.cos(V_GRID * 205.0 + np.sin(U_GRID * 18.0) * 1.2)
        sphere_3d = np.sqrt(np.clip(1.0 - (pr / (p_rad + 1e-4)) ** 2, 0.0, 1.0))
        planet = (
            np.array([0.88, 0.34, 0.46], np.float32)
            * (p_mask * bands * (0.3 + 0.7 * sphere_3d))[..., None]
        )

        eyeball = eyeball * (1.0 - pmask)[..., None] + planet * (pmask * sp)[..., None]

    col = np.where(inside[..., None], eyeball, skin)
    return np.clip(col, 0.0, 1.0), (inside & (er < rp))


def main():
    root = tk.Tk()
    root.title("Eye -> Space")
    root.configure(bg="black")
    root.resizable(False, False)

    canvas = tk.Canvas(
        root, width=WIDTH, height=HEIGHT, bg="black", highlightthickness=0
    )
    canvas.pack()

    font_lut = build_font_lut()

    with open(__file__, "r", encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f if line.strip()]

    base_code_idx = np.zeros((ROWS, COLS), dtype=np.int32)
    for r in range(ROWS):
        line = raw_lines[r % len(raw_lines)]
        while len(line) < COLS:
            line += "   " + raw_lines[(r + len(line)) % len(raw_lines)]
        for c, ch in enumerate(line[:COLS]):
            code = ord(ch) if ord(ch) < 128 else ord(":")
            if font_lut[code].sum() == 0 and ch != " ":
                code = ord(":")
            base_code_idx[r, c] = code

    rng = np.random.RandomState(42)
    star_coords = [
        (rng.randint(4, ROWS - 4), rng.randint(4, COLS - 4), rng.choice([1, 2]))
        for _ in range(38)
    ]

    ppm_header = f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii")
    start_time = time.time()

    def update_frame():
        t0 = time.perf_counter()
        t = (time.time() - start_time) % 15.0

        oa = 1.0
        if 1.15 < t < 1.55:
            oa = np.abs((t - 1.35) / 0.20)
        elif 4.35 < t < 4.75:
            oa = np.abs((t - 4.55) / 0.20)

        gx, gy = 0.0, 0.0
        if 1.6 <= t < 2.9:
            gx = -0.055 * np.sin((t - 1.6) / 1.3 * np.pi)
        elif 2.9 <= t < 4.2:
            gx = 0.055 * np.sin((t - 2.9) / 1.3 * np.pi)

        if t < 6.0:
            rp = 0.088
            sp = 0.0
        else:
            z = np.clip((t - 6.0) / 5.2, 0.0, 1.0)
            ease = z * z * (3.0 - 2.0 * z)
            rp = 0.088 + ease * 0.95
            oa = 1.0 + ease * 1.8
            sp = np.clip((t - 7.2) / 3.2, 0.0, 1.0)

        rgb, in_pupil = render_eye(oa, gx, gy, rp, sp)
        luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

        # Vectorized character selection
        char_idx = base_code_idx.copy()
        spaces = char_idx == ord(" ")
        char_idx[spaces & (luma < 0.35)] = ord(":")
        char_idx[spaces & (luma >= 0.35)] = ord("#")
        char_idx[luma <= 0.025] = ord(" ")

        if sp > 0.05:
            s_col = np.array([sp, sp, sp], dtype=np.float32)
            for sr, sc, sz in star_coords:
                if in_pupil[sr, sc] and luma[sr, sc] < 0.15:
                    pts = [
                        (sr, sc, "+"),
                        (sr, sc - 1, "-"),
                        (sr, sc + 1, "-"),
                        (sr - 1, sc, "|"),
                        (sr + 1, sc, "|"),
                    ]
                    if sz == 2:
                        pts.extend([(sr, sc - 2, "."), (sr, sc + 2, ".")])
                    for r_p, c_p, ch_p in pts:
                        char_idx[r_p, c_p] = ord(ch_p)
                        rgb[r_p, c_p] = s_col

        # Vectorized GPU-style tile blitting in NumPy (~2ms)
        glyphs = font_lut[char_idx]  # (ROWS, COLS, CH, CW)
        frame_rgb = (glyphs[..., None] * rgb[:, :, None, None, :] * 255.0).astype(
            np.uint8
        )
        img_bytes = (
            frame_rgb.transpose(0, 2, 1, 3, 4).reshape(HEIGHT, WIDTH, 3).tobytes()
        )

        photo = tk.PhotoImage(
            width=WIDTH, height=HEIGHT, data=ppm_header + img_bytes, format="PPM"
        )
        canvas.image = photo
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor="nw")

        # Dynamic frame pacing for locked 60 FPS (16ms per frame)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        root.after(max(1, 16 - elapsed_ms), update_frame)

    update_frame()
    root.mainloop()


if __name__ == "__main__":
    main()