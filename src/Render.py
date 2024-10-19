import numpy as np
import cupy as cp
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import ctypes
import freetype

# Vertex shader for particles
PARTICLE_VERTEX_SHADER = """
#version 330
in vec2 position;
in float particleType;
out float vParticleType;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    vParticleType = particleType;
}
"""

# Fragment shader for particles
PARTICLE_FRAGMENT_SHADER = """
#version 330
in float vParticleType;
out vec4 fragColor;

vec3 getColor(float type) {
    if (type == 0.0) return vec3(0.0, 0.0, 0.0);  // Black
    else if (type == 1.0) return vec3(0.0, 1.0, 0.0);  // Green
    else if (type == 2.0) return vec3(1.0, 0.5, 0.0);  // Orange
    else if (type == 3.0) return vec3(1.0, 1.0, 0.0);  // Yellow
    else return vec3(1.0, 1.0, 1.0);  // White (default)
}

void main() {
    fragColor = vec4(getColor(vParticleType), 1.0);
}
"""

# Vertex shader for text
TEXT_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>
out vec2 TexCoords;

uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
    TexCoords = vertex.zw;
}
"""

# Fragment shader for text
TEXT_FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoords;
out vec4 color;

uniform sampler2D text;
uniform vec3 textColor;

void main()
{    
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
    color = vec4(textColor, 1.0) * sampled;
}
"""

# Update the heatmap vertex shader
HEATMAP_VERTEX_SHADER = """
#version 330
in vec2 position;
in float intensity;
out float vIntensity;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    vIntensity = intensity;
}
"""

# The fragment shader remains the same
HEATMAP_FRAGMENT_SHADER = """
#version 330
in float vIntensity;
out vec4 fragColor;

vec3 getHeatmapColor(float value) {
    value = clamp(value, 0.0, 1.0);
    return vec3(value, 0.0, 1.0 - value);  // Blue to red gradient
}

void main() {
    fragColor = vec4(getHeatmapColor(vIntensity), 1.0);
}
"""

LINE_PLOT_VERTEX_SHADER = """
#version 330
in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

LINE_PLOT_FRAGMENT_SHADER = """
#version 330
uniform vec3 lineColor;
out vec4 fragColor;
void main() {
    fragColor = vec4(lineColor, 1.0);
}
"""

class CharacterSlot:
    def __init__(self, texture, glyph):
        self.texture = texture
        self.textureSize = (glyph.bitmap.width, glyph.bitmap.rows)

        if isinstance(glyph, freetype.GlyphSlot):
            self.bearing = (glyph.bitmap_left, glyph.bitmap_top)
            self.advance = glyph.advance.x
        elif isinstance(glyph, freetype.BitmapGlyph):
            self.bearing = (glyph.left, glyph.top)
            self.advance = None
        else:
            raise RuntimeError('unknown glyph type')

def _get_rendering_buffer(xpos, ypos, w, h, zfix=0.0):
    return np.asarray([
        xpos,     ypos - h, 0, 0,
        xpos,     ypos,     0, 1,
        xpos + w, ypos,     1, 1,
        xpos,     ypos - h, 0, 0,
        xpos + w, ypos,     1, 1,
        xpos + w, ypos - h, 1, 0
    ], np.float32)

def ortho_matrix(left, right, bottom, top, near, far):
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2 / (right - left)
    m[1, 1] = 2 / (top - bottom)
    m[2, 2] = -2 / (far - near)
    m[3, 0] = -(right + left) / (right - left)
    m[3, 1] = -(top + bottom) / (top - bottom)
    m[3, 2] = -(far + near) / (far - near)
    m[3, 3] = 1
    return m

class PICRenderer:
    def __init__(self, width, height, fontfile, renderer_type="particles"):
        self.width = width
        self.height = height
        self.fontfile = fontfile
        self.renderer_type = renderer_type  # Choose either 'particles' or 'heatmap'
        self.text_content = {}
        print(f'Initializing {renderer_type} PIC Renderer...')
        
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        self.window = glfw.create_window(width, height, "Multi-type PIC Renderer with Text (CuPy)", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.make_context_current(self.window)

        # Initialize rendering based on renderer type
        '''
        if self.renderer_type == "particles":
            self.init_particle_rendering()
        elif self.renderer_type == "heatmap":
            self.init_heatmap_rendering()
        elif self.renderer_type == "line_plot":
            self.init_line_plot_rendering()
        '''

        self.init_particle_rendering()
        self.init_heatmap_rendering()
        self.init_line_plot_rendering()
        # Initialize text rendering
        self.init_text_rendering()

        glfw.swap_interval(0)

        self.line_plots = {}  # Dictionary to store multiple line plots
        self.line_colors = {}  # Dictionary to store colors for each line plot
        self.num_line_points = 0  # Initialize num_line_pointss

    def set_renderer_type(self, renderer_type):
        """Change the rendering mode between particles and heatmap."""
        self.renderer_type = renderer_type
        if self.renderer_type == "particles":
            self.init_particle_rendering()
        elif self.renderer_type == "heatmap":
            self.init_heatmap_rendering()

    # Other methods...


    def init_particle_rendering(self):
        vertex_shader = shaders.compileShader(PARTICLE_VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(PARTICLE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.particle_shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        
        self.particle_vao = glGenVertexArrays(1)
        glBindVertexArray(self.particle_vao)
        
        self.particle_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
        
        position_attrib = glGetAttribLocation(self.particle_shader_program, "position")
        glEnableVertexAttribArray(position_attrib)
        glVertexAttribPointer(position_attrib, 2, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        
        type_attrib = glGetAttribLocation(self.particle_shader_program, "particleType")
        glEnableVertexAttribArray(type_attrib)
        glVertexAttribPointer(type_attrib, 1, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(8))
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def init_text_rendering(self):
        vertex_shader = shaders.compileShader(TEXT_VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(TEXT_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.text_shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        
        projection = ortho_matrix(0, self.width, self.height, 0, -1, 1)
        glUseProgram(self.text_shader_program)
        shader_projection = glGetUniformLocation(self.text_shader_program, "projection")
        glUniformMatrix4fv(shader_projection, 1, GL_FALSE, projection)

        self.Characters = {}
        face = freetype.Face(self.fontfile)
        face.set_char_size(48*64)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        for i in range(0, 128):
            face.load_char(chr(i))
            glyph = face.glyph
            
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, glyph.bitmap.width, glyph.bitmap.rows, 0,
                         GL_RED, GL_UNSIGNED_BYTE, glyph.bitmap.buffer)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            self.Characters[chr(i)] = CharacterSlot(texture, glyph)
        
        glBindTexture(GL_TEXTURE_2D, 0)

        self.text_vao = glGenVertexArrays(1)
        glBindVertexArray(self.text_vao)
        
        self.text_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.text_vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 4 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def init_heatmap_rendering(self):
        vertex_shader = shaders.compileShader(HEATMAP_VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(HEATMAP_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.heatmap_shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        
        self.heatmap_vao = glGenVertexArrays(1)
        glBindVertexArray(self.heatmap_vao)

        self.heatmap_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.heatmap_vbo)

        position_attrib = glGetAttribLocation(self.heatmap_shader_program, "position")
        glEnableVertexAttribArray(position_attrib)
        glVertexAttribPointer(position_attrib, 2, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

        intensity_attrib = glGetAttribLocation(self.heatmap_shader_program, "intensity")
        glEnableVertexAttribArray(intensity_attrib)
        glVertexAttribPointer(intensity_attrib, 1, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(8))
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def init_line_plot_rendering(self):
        """Initialize the rendering pipeline for line plots."""
        """Initialize the rendering pipeline for line plots."""
        vertex_shader = shaders.compileShader(LINE_PLOT_VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(LINE_PLOT_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.line_plot_shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        
        self.line_vao = glGenVertexArrays(1)
        glBindVertexArray(self.line_vao)
        
        self.line_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)

        position_attrib = glGetAttribLocation(self.line_plot_shader_program, "position")
        glEnableVertexAttribArray(position_attrib)
        glVertexAttribPointer(position_attrib, 2, GL_FLOAT, GL_FALSE, 0, None)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def update_particles(self, particle_positions, particle_types):
        assert particle_positions.shape[0] == 2, "Position shape should be (2, n)"
        assert particle_positions.shape[1] == particle_types.shape[0], "Number of positions and types should match"
        
        gl_positions = 2.0 * particle_positions - 1.0
        gl_positions = cp.ascontiguousarray(gl_positions.T).astype(cp.float32)
        
        particle_data = cp.column_stack((gl_positions, particle_types.astype(cp.float32)))
        
        particle_data_cpu = particle_data.get()
        
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
        glBufferData(GL_ARRAY_BUFFER, particle_data_cpu.nbytes, particle_data_cpu, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        self.num_particles = gl_positions.shape[0]


    def update_text(self, key, content):
        """
        Update or add a text entry to be rendered.
        
        :param key: A unique identifier for this text (e.g., 'frame_counter', 'particle_count')
        :param content: A dictionary containing text properties:
                        {
                            'text': str,  # The text to display
                            'x': int,     # X-coordinate position
                            'y': int,     # Y-coordinate position
                            'scale': float,  # Text scale
                            'color': tuple   # RGB color tuple (0-255 for each component)
                        }
        """
        self.text_content[key] = content

    def reshape_cupy_1d_to_2d(self, data_1d, rows, cols):
        # Reshape CuPy 1D data into 2D grid
        return cp.reshape(data_1d, (rows, cols))
    
    def update_heatmap(self, data_1d_cupy, rows, cols):
        # Reshape the CuPy 1D data to 2D
        heatmap_data_cupy = data_1d_cupy.reshape((rows, cols))
        
        # Normalize intensity values between 0 and 1
        heatmap_min = cp.min(heatmap_data_cupy)
        heatmap_max = cp.max(heatmap_data_cupy)
        heatmap_data_cupy_normalized = (heatmap_data_cupy - heatmap_min) / (heatmap_max - heatmap_min)
        
        # Create vertex data for quads using CuPy operations
        x = cp.linspace(-1, 1, cols)
        y = cp.linspace(-1, 1, rows)
        X, Y = cp.meshgrid(x, y)
        
        # Create vertices for two triangles per grid cell
        vertices = cp.zeros((rows-1, cols-1, 6, 3), dtype=cp.float32)
        
        vertices[:,:,0,0] = X[:-1,:-1]  # x1
        vertices[:,:,0,1] = Y[:-1,:-1]  # y1
        vertices[:,:,0,2] = heatmap_data_cupy_normalized[:-1,:-1]  # i1
        
        vertices[:,:,1,0] = X[:-1,1:]   # x2
        vertices[:,:,1,1] = Y[:-1,:-1]  # y1
        vertices[:,:,1,2] = heatmap_data_cupy_normalized[:-1,1:]  # i2
        
        vertices[:,:,2,0] = X[1:,:-1]   # x1
        vertices[:,:,2,1] = Y[1:,:-1]   # y2
        vertices[:,:,2,2] = heatmap_data_cupy_normalized[1:,:-1]  # i3
        
        vertices[:,:,3,0] = X[1:,:-1]   # x1
        vertices[:,:,3,1] = Y[1:,:-1]   # y2
        vertices[:,:,3,2] = heatmap_data_cupy_normalized[1:,:-1]  # i3
        
        vertices[:,:,4,0] = X[:-1,1:]   # x2
        vertices[:,:,4,1] = Y[:-1,:-1]  # y1
        vertices[:,:,4,2] = heatmap_data_cupy_normalized[:-1,1:]  # i2
        
        vertices[:,:,5,0] = X[1:,1:]    # x2
        vertices[:,:,5,1] = Y[1:,:-1]   # y2
        vertices[:,:,5,2] = heatmap_data_cupy_normalized[1:,1:]   # i4
        
        # Reshape to a 2D array of vertices
        vertices_cupy = vertices.reshape(-1, 3)
        
        # Transfer data from GPU (CuPy) to CPU (NumPy) for OpenGL
        vertices_cpu = cp.asnumpy(vertices_cupy)

        # Update the OpenGL buffer with CPU data
        glBindBuffer(GL_ARRAY_BUFFER, self.heatmap_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices_cpu.nbytes, vertices_cpu, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        # Update the number of vertices to render
        self.num_heatmap_vertices = vertices_cpu.shape[0]

    def update_line_data(self, plot_id, x_values, y_values, color=(1.0, 1.0, 1.0)):
        """Update or add a line plot dataset."""
        assert len(x_values) == len(y_values), "x and y arrays must have the same length"

        # Ensure x_values and y_values are CuPy arrays
        x_values = cp.asarray(x_values)
        y_values = cp.asarray(y_values)

        # Normalize coordinates to OpenGL's -1.0 to 1.0 range
        x_normalized = 2.0 * (x_values - cp.min(x_values)) / (cp.max(x_values) - cp.min(x_values)) - 1.0
        y_normalized = 2.0 * (y_values - cp.min(y_values)) / (cp.max(y_values) - cp.min(y_values)) - 1.0

        # Combine into vertex array
        vertices = cp.column_stack((x_normalized, y_normalized)).astype(cp.float32)

        # Transfer data from GPU (CuPy) to CPU (NumPy) for OpenGL
        vertices_cpu = cp.asnumpy(vertices)

        # Create a new VBO if this is a new plot_id
        if plot_id not in self.line_plots:
            vbo = glGenBuffers(1)
            self.line_plots[plot_id] = {'vbo': vbo, 'num_points': 0}

        # Update the buffer for this plot_id
        glBindBuffer(GL_ARRAY_BUFFER, self.line_plots[plot_id]['vbo'])
        glBufferData(GL_ARRAY_BUFFER, vertices_cpu.nbytes, vertices_cpu, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self.line_plots[plot_id]['num_points'] = len(x_values)
        self.line_colors[plot_id] = color

    def render_particles(self):
        glUseProgram(self.particle_shader_program)
        glBindVertexArray(self.particle_vao)
        glDrawArrays(GL_POINTS, 0, self.num_particles)
        glBindVertexArray(0)

    def render_text(self, text, x, y, scale, color):
        glUseProgram(self.text_shader_program)
        glUniform3f(glGetUniformLocation(self.text_shader_program, "textColor"),
                    color[0]/255, color[1]/255, color[2]/255)
        
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(self.text_vao)
        for c in text:
            ch = self.Characters[c]
            w, h = ch.textureSize
            w = w * scale
            h = h * scale
            vertices = _get_rendering_buffer(x, y, w, h)

            glBindTexture(GL_TEXTURE_2D, ch.texture)
            glBindBuffer(GL_ARRAY_BUFFER, self.text_vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            x += (ch.advance >> 6) * scale

        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)

    def render_heatmap(self):
        glUseProgram(self.heatmap_shader_program)
        glBindVertexArray(self.heatmap_vao)
        glDrawArrays(GL_TRIANGLES, 0, self.num_heatmap_vertices)
        glBindVertexArray(0)

    def render_line_plots(self):
        """Render all line plots."""
        glUseProgram(self.line_plot_shader_program)
        
        for plot_id, plot_data in self.line_plots.items():
            glBindBuffer(GL_ARRAY_BUFFER, plot_data['vbo'])
            position_attrib = glGetAttribLocation(self.line_plot_shader_program, "position")
            glEnableVertexAttribArray(position_attrib)
            glVertexAttribPointer(position_attrib, 2, GL_FLOAT, GL_FALSE, 0, None)

            # Set the color for this line plot
            color_loc = glGetUniformLocation(self.line_plot_shader_program, "lineColor")
            glUniform3f(color_loc, *self.line_colors[plot_id])

            glDrawArrays(GL_LINE_STRIP, 0, plot_data['num_points'])
            
            glDisableVertexAttribArray(position_attrib)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render(self, clear=True):
        if clear:
            glClear(GL_COLOR_BUFFER_BIT)
            glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background

        if self.renderer_type == "particles":
            self.render_particles()
        elif self.renderer_type == "heatmap":
            self.render_heatmap()
        elif self.renderer_type == "line_plot":
            self.render_line_plots()
        # Render all text entries
        for content in self.text_content.values():
            self.render_text(content['text'], content['x'], content['y'], content['scale'], content['color'])

        glfw.swap_buffers(self.window)
        glfw.poll_events()
    
    def should_close(self):
        return glfw.window_should_close(self.window)

    def close(self):
        glfw.terminate()

# Example usage
if __name__ == "__main__":
    width, height = 800, 600
    fontfile = "C:\\Windows\\Fonts\\Arial.ttf"  # Adjust this path as needed
    renderer = PICRenderer(width, height, fontfile)
    
    n = 10000
    particle_positions = cp.random.rand(2, n).astype(cp.float32)
    particle_types = cp.random.randint(0, 4, n).astype(cp.float32)
    
    frame = 0
    start_time = glfw.get_time()
    while not renderer.should_close():
        # Update particle positions (simple circular motion)
        t = frame * 0.01
        particle_positions[0] = cp.cos(t + particle_positions[0] * 2 * cp.pi) * 0.5 + 0.5
        particle_positions[1] = cp.sin(t + particle_positions[1] * 2 * cp.pi) * 0.5 + 0.5
        
        renderer.update_particles(particle_positions, particle_types)
        
        # Update text content
        renderer.update_text('frame_counter', {
            'text': f"Frame: {frame}",
            'x': 10,
            'y': 30,
            'scale': 0.5,
            'color': (255, 255, 255)
        })
        
        renderer.update_text('particle_count', {
            'text': f"Particles: {n}",
            'x': 10,
            'y': 60,
            'scale': 0.5,
            'color': (200, 200, 0)
        })
        
        # Calculate and display FPS
        current_time = glfw.get_time()
        fps = frame / (current_time - start_time)
        renderer.update_text('fps', {
            'text': f"FPS: {fps:.2f}",
            'x': width - 150,
            'y': 30,
            'scale': 0.5,
            'color': (0, 255, 0)
        })
        
        renderer.render()
        frame += 1
    
    renderer.close()