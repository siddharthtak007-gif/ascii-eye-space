import time
import numpy as np
import pygame

# Grid resolution (Columns x Rows of ASCII characters)
COLS, ROWS = 130, 55
FONT_SIZE = 13
CHAR_W, CHAR_H = 7, 13
WIDTH, HEIGHT = COLS * CHAR_W, ROWS * CHAR_H

# ASCII ramp from darkest to brightest
ASCII_CHARS = " .:-=+*#%@M$"


def ss(e0, e1, x):
    """Smoothstep function for smooth mathematical transitions"""
    t = np.clip((x - e0) / (e1 - e0 + 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def noise(x, y):
    """2D pseudo-random procedural noise"""
    n = np.sin(x * 12.9898 + y * 78.233) * 43758.5453
    return n - np.floor(n)


def render_eye(u, v, open_amt, gx, gy, rp, sp):
    """Exact procedural eye shader reconstructed from the video's code"""
    # Aspect-corrected centered coordinates
    x = (u - 0.5) * 2.2
    y = (v - 0.5) * 1.4

    # Eyelid curves (upper and lower)
    r_pos = 0.36 * open_amt * (1.0 - 0.85 * (x * x))
    r_neg = -0.28 * open_amt * (1.0 - 0.85 * (x * x))

    upper = y >= 0
    du = np.abs(y - r_pos)
    dl = np.abs(y - r_neg)
    d_lid = np.where(upper, du, dl)
    inside = (y < r_pos) & (y > r_neg)

    # 1. Skin & Eyelid Shading
    tx = noise(u * 120 + 100, v * 120 + 100)
    skin_base = np.array([0.82, 0.48, 0.29], dtype=np.float32)
    skin = skin_base * (0.90 + 0.20 * tx)[..., None]

    # Eyelid crease & shadow
    ou = ss(0.0, 0.35, du)
    cr = np.exp(-((y - 0.28) / 0.09) ** 2) * (1.0 - np.clip(x * x * 1.5, 0, 1))
    lid_tone = (
        0.55
        + 0.40 * ou
        - 0.30 * cr * upper
        + 0.15 * np.exp(-((du - r_pos) / 0.028) ** 2) * upper
        - 0.25 * np.exp(-((dl - np.abs(r_neg)) / 0.04) ** 2) * (~upper)
    )

    sh = ss(0.0, 0.05, d_lid)
    skin = skin * lid_tone[..., None] * (0.4 + 0.6 * sh)[..., None]
    skin = skin + np.array([0.08, 0.03, 0.01], dtype=np.float32) * (1.0 - sh)[
        ..., None
    ] * (lid_tone > 0.5)[..., None]

    # 2. Eyebrow
    by = y - (0.44 - 0.10 * (x * x) + 0.04 * x)
    eb = np.exp(-((by / (0.048 + 0.012 * np.abs(x))) ** 2)) * ss(
        0.75, 0.25, np.abs(x)
    )
    hair = np.sin(u * 250 + v * 80) * 0.5 + 0.5
    eb_val = np.clip(eb * (0.6 + 0.4 * hair), 0.0, 0.92)[..., None]
    skin = skin * (1.0 - eb_val) + np.array(
        [0.12, 0.07, 0.04], dtype=np.float32
    ) * eb_val

    # 3. Eyelashes
    lash_freq = np.sin(u * 280.0) * 0.5 + 0.5
    ul = (
        np.exp(-((du / 0.022) ** 2))
        * upper
        * (0.4 + 0.6 * lash_freq)
        * ss(0.65, 0.15, np.abs(x))
    )
    ll = (
        np.exp(-((dl / 0.015) ** 2))
        * (~upper)
        * (0.3 + 0.5 * lash_freq)
        * ss(0.60, 0.15, np.abs(x))
    )
    skin = skin * (1.0 - np.clip(ul + ll, 0, 0.92)[..., None])

    # 4. Eyeball (Sclera, Iris, Pupil, Reflections)
    ex = x - gx
    ey = y - gy
    er = np.sqrt(ex * ex + ey * ey)
    etheta = np.arctan2(ey, ex)

    # Sclera (white of the eye with subtle blood vessels & shading)
    sclera_shade = np.clip(1.0 - 0.65 * (er / 0.65) ** 2, 0.25, 1.0)
    veins = np.sin(etheta * 25.0 + er * 40.0) * 0.5 + 0.5
    sclera = np.array([0.88, 0.82, 0.78], dtype=np.float32) * sclera_shade[..., None]
    sclera -= (
        np.array([0.0, 0.12, 0.15], dtype=np.float32)
        * (veins * ss(0.28, 0.55, er))[..., None]
        * 0.3
    )

    # Iris (golden/brown radial fibers)
    r_iris = 0.27
    iris_mask = ss(r_iris, r_iris - 0.02, er)
    fibers = (
        np.sin(etheta * 45.0 + np.sin(er * 35.0) * 2.0) * 0.5
        + 0.5
        + np.cos(etheta * 90.0) * 0.25
    )
    iris_inner = ss(r_iris, rp, er)
    iris_col = (
        np.array([0.78, 0.45, 0.16], dtype=np.float32) * (0.55 + 0.65 * fibers)[..., None]
        + np.array([0.95, 0.68, 0.25], dtype=np.float32) * (iris_inner * 0.35)[..., None]
    )
    limbal_ring = ss(r_iris - 0.04, r_iris, er)
    iris_col *= (1.0 - 0.75 * limbal_ring)[..., None]

    # Pupil & Specular Highlight
    pupil_mask = ss(rp, rp - 0.015, er)
    spec1 = np.exp(-(((ex - 0.08) ** 2 + (ey - 0.08) ** 2) / 0.0018))
    spec2 = 0.4 * np.exp(-(((ex + 0.09) ** 2 + (ey + 0.06) ** 2) / 0.0008))
    spec = np.clip(spec1 + spec2, 0.0, 1.0) * (1.0 - sp)

    eye_col = sclera * (1.0 - iris_mask)[..., None] + iris_col * iris_mask[..., None]
    eye_col = eye_col * (1.0 - pupil_mask)[..., None]
    eye_col += np.array([1.0, 0.98, 0.95], dtype=np.float32) * spec[..., None]

    # Upper eyelid shadow cast onto eyeball
    eye_shadow = ss(0.0, 0.07, du) * ss(0.0, 0.04, dl)
    eye_col *= (0.35 + 0.65 * eye_shadow)[..., None]

    # 5. Cosmic Space / Nebula inside the expanding pupil
    if sp > 0.001:
        sx = ex / (rp + 1e-4)
        sy = ey / (rp + 1e-4)
        sr = np.sqrt(sx * sx + sy * sy)

        # Glowing pink/magenta central core (matches 0:11 - 0:14 in video)
        neb_mask = np.exp(-((sr / 0.42) ** 2.2))
        neb_stripes = 0.65 + 0.35 * np.sin(v * 160.0 + u * 20.0)
        neb_col = (
            np.array([0.95, 0.28, 0.48], dtype=np.float32)
            * (neb_mask * neb_stripes)[..., None]
        )

        # Starfield with 4-point cross flares
        star_grid = noise(np.floor(u * 45.0), np.floor(v * 25.0))
        stars = (star_grid > 0.978).astype(np.float32)
        cross_h = (
            (noise(np.floor(u * 45.0 - 1), np.floor(v * 25.0)) > 0.988)
            | (noise(np.floor(u * 45.0 + 1), np.floor(v * 25.0)) > 0.988)
        ).astype(np.float32) * 0.5
        cross_v = (
            (noise(np.floor(u * 45.0), np.floor(v * 25.0 - 1)) > 0.988)
            | (noise(np.floor(u * 45.0), np.floor(v * 25.0 + 1)) > 0.988)
        ).astype(np.float32) * 0.5

        star_brightness = np.clip(stars + cross_h + cross_v, 0.0, 1.0) * (
            1.0 - neb_mask * 0.7
        )
        space_col = (
            neb_col
            + np.array([0.95, 0.95, 1.0], dtype=np.float32) * star_brightness[..., None]
        )

        eye_col = eye_col * (1.0 - pupil_mask)[..., None] + space_col * (
            pupil_mask * sp
        )[..., None]

    # Combine Eyeball and Skin
    final_col = np.where(inside[..., None], eye_col, skin)
    return np.clip(final_col, 0.0, 1.0)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Eye -> Space")
    font = pygame.font.SysFont("consolas", FONT_SIZE, bold=True)
    clock = pygame.time.Clock()

    # Pre-render ASCII characters for fast performance
    u_grid, v_grid = np.meshgrid(
        np.linspace(0, 1, COLS, dtype=np.float32),
        np.linspace(0, 1, ROWS, dtype=np.float32),
    )

    start_time = time.time()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 15-second looping timeline matching the video
        t = (time.time() - start_time) % 15.0

        # Animation keyframes: Blink -> Look Left/Right -> Zoom into Space
        open_amt = 1.0
        if 1.1 < t < 1.55:
            open_amt = np.abs((t - 1.325) / 0.225)
        elif 4.4 < t < 4.85:
            open_amt = np.abs((t - 4.625) / 0.225)

        gx, gy = 0.0, 0.0
        if 1.6 <= t < 3.0:
            gx = -0.09 * np.sin((t - 1.6) / 1.4 * np.pi)
        elif 3.0 <= t < 4.3:
            gx = 0.09 * np.sin((t - 3.0) / 1.3 * np.pi)

        if t < 6.0:
            rp = 0.095
            sp = 0.0
        else:
            progress = np.clip((t - 6.0) / 5.5, 0.0, 1.0)
            ease = progress * progress * (3.0 - 2.0 * progress)
            rp = 0.095 + ease * 1.35
            open_amt = 1.0 + ease * 2.5
            sp = np.clip((t - 7.5) / 3.0, 0.0, 1.0)

        # Compute RGB frame
        rgb = render_eye(u_grid, v_grid, open_amt, gx, gy, rp, sp)
        luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        char_indices = (luma * (len(ASCII_CHARS) - 1)).astype(np.int32)
        colors = (rgb * 255).astype(np.uint8)

        # Draw frame
        screen.fill((0, 0, 0))
        for r in range(ROWS):
            y_pos = r * CHAR_H
            for c in range(COLS):
                ch_idx = char_indices[r, c]
                if ch_idx > 0:
                    ch = ASCII_CHARS[ch_idx]
                    col = tuple(colors[r, c])
                    surf = font.render(ch, True, col)
                    screen.blit(surf, (c * CHAR_W, y_pos))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()