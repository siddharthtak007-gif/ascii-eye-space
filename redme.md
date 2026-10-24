Procedural ASCII Eye Space Shader
This project is a real-time procedural ASCII shader written entirely in Python. It mathematically generates an animated eye that blinks, looks around, and eventually zooms through the pupil into a glowing pink nebula and starry cosmos—rendered entirely in ASCII characters at a smooth 60 FPS.

What We Made
A self-contained, 60 FPS ASCII rendering engine. Instead of using pre-rendered video or external 3D engines, this script uses mathematical formulas (like fractional Brownian motion and smoothstep) to calculate skin tones, shadows, iris fibers, and planetary rings on the fly. As a bonus, the "texture" of the eye and the planet is actually the source code of the Python file itself.

What We Use
Python 3.x: The core programming language.

NumPy: Used for heavy matrix math, vectorized array calculations, and fast image manipulation.

Tkinter: Python's built-in GUI library. We use its Canvas widget to display the final frame.

Time: For syncing the animation keyframes (blinking, zooming) to a 15-second loop.

How It Works
Mathematical Shading: Everything on screen is drawn using math. The script uses procedural noise (fbm) for the skin texture and eyebrow hair, and distance fields (er = np.sqrt(ex*ex + ey*ey)) to draw the circular iris and pupil.

Vectorized Processing: Instead of calculating the screen pixel by pixel (which is slow), numpy calculates the entire grid simultaneously using 2D coordinate grids (U_GRID, V_GRID).

Self-Referential ASCII: The script reads its own __file__ (the source code) and maps those characters into a grid.

PPM Framebuffer Engine: To achieve 60 FPS, standard UI text widgets are too slow. Instead, the script uses a custom-encoded 5x9 hex font. NumPy maps the calculated colors and ASCII characters to this font, generates a raw binary PPM image frame in about 2 milliseconds, and pushes it directly to the Tkinter canvas.

Requirements Needed
Python 3.6 or higher installed on your system.

The numpy library.

Note: tkinter comes pre-installed with standard Python distributions.

How to Run It (Terminal Commands)
Clone or download this repository and navigate to the folder containing the file.

Install the required numpy dependency by running this in your terminal:

Bash
python -m pip install numpy
Run the script:

Bash
python eye_space.py

python -m pip install numpy

python eye_space.py