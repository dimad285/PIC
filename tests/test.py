import numpy as np
from vispy import app, gloo

# Initialize the particle data
n_particles = 100000
positions = np.random.rand(n_particles, 2).astype(np.float32)  # 2D positions
colors = np.random.rand(n_particles, 4).astype(np.float32)  # RGBA colors

# OpenGL shaders for rendering points
vertex_shader = """
attribute vec2 a_position;
attribute vec4 a_color;
varying vec4 v_color;

void main() {
    v_color = a_color;
    gl_Position = vec4(a_position, 0.0, 1.0);
    gl_PointSize = 2.0;
}
"""

fragment_shader = """
varying vec4 v_color;

void main() {
    gl_FragColor = v_color;
}
"""

# Create a canvas and program
class ParticleCanvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(800, 800))
        self.program = gloo.Program(vertex_shader, fragment_shader)

        # Bind particle data to shaders
        self.program['a_position'] = positions
        self.program['a_color'] = colors

        # OpenGL settings
        gloo.set_state(clear_color='black', blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))

        self.timer = app.Timer('auto', connect=self.update, start=True)

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('points')  # Render points

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)

# Run the canvas
canvas = ParticleCanvas()
canvas.show()
app.run()