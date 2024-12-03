import numpy as np
import cupy as cp
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import ctypes
import freetype
import glm

# Vertex shader for particles
PARTICLE_VERTEX_SHADER_2D = """
#version 330
in vec2 position;
in float particleType;
out float vParticleType;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    vParticleType = particleType;
}
"""

PARTICLE_VERTEX_SHADER_3D = """
#version 330 core

layout (location = 0) in vec3 position;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main() {
    vec4 worldPosition = model * vec4(position, 1.0);
    gl_Position = projection * view * worldPosition;
    gl_PointSize = 5.0;  // Set particle size
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


PARTICLE_FRAGMENT_SHADER_3D = """
#version 330 core

out vec4 fragColor;

void main() {
    fragColor = vec4(1.0, 1.0, 1.0, 1.0);  // White color for particles
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


SURFACE_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPosition;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

out float vertexZ;  // Pass the Z-coordinate to the fragment shader

void main() {
    gl_Position = projection * view * model * vec4(aPosition, 1.0);
    vertexZ = aPosition.z;  // Pass the Z-coordinate
}
"""

SURFACE_FRAGMENT_SHADER = """
#version 330 core

in float vertexZ;  // Receive the Z-coordinate from the vertex shader
out vec4 fragColor;

uniform float zMin;  // Minimum Z value (to normalize)
uniform float zMax;  // Maximum Z value (to normalize)

void main() {
    float normalizedZ = (vertexZ - zMin) / (zMax - zMin);
    vec3 lowColor = vec3(0.0, 0.0, 1.0);  // Blue
    vec3 highColor = vec3(1.0, 0.0, 0.0);  // Red
    vec3 finalColor = mix(lowColor, highColor, normalizedZ);
    fragColor = vec4(finalColor, 1.0);  // Final color with full opacity
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

def get_projection_matrix(fov, aspect, near, far):
    """Create a perspective projection matrix."""
    f = 1.0 / np.tan(fov / 2)
    proj = np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), -1],
        [0, 0, (2 * far * near) / (near - far), 0]
    ], dtype=np.float32)
    return proj

def get_view_matrix(eye, center, up):
    """Create a view matrix for the camera."""
    f = (center - eye) / np.linalg.norm(center - eye)
    s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
    u = np.cross(s, f)
    
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[0, 3] = -np.dot(s, eye)
    view[1, 3] = -np.dot(u, eye)
    view[2, 3] = np.dot(f, eye)
    
    return view

class PICRenderer:
    def __init__(self, width, height, fontfile, renderer_type="particles", is_3d = False):
        self.width = width
        self.height = height
        self.fontfile = fontfile
        self.renderer_type = renderer_type  # Choose either 'particles' or 'heatmap'
        self.text_content = {}
        self.is_3d = is_3d
        self.line_plot_type = None
        self.fov = np.radians(60)
        self.label_list = []
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
        self.init_surface_rendering()
        # Initialize text rendering
        self.init_text_rendering()

        glfw.swap_interval(0)

        self.line_plots = {}  # Dictionary to store multiple line plots
        self.line_colors = {}  # Dictionary to store colors for each line plot
        self.num_line_points = 0  # Initialize num_line_pointss

    def set_renderer_type(self, renderer_type, is_3d=False):
        """Set the renderer type and whether it's 2D or 3D."""
        self.renderer_type = renderer_type
        self.is_3d = is_3d  # Boolean to control 2D or 3D mode
        self.init_particle_rendering(is_3d=is_3d)

    def set_fov(self, fov):
        self.fov = fov

    def set_camera(self, eye):
        self.eye = eye

    def init_particle_rendering(self, is_3d=False):
        if is_3d:
            vertex_shader = shaders.compileShader(PARTICLE_VERTEX_SHADER_3D, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(PARTICLE_FRAGMENT_SHADER_3D, GL_FRAGMENT_SHADER)
            self.particle_shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        else:
            vertex_shader = shaders.compileShader(PARTICLE_VERTEX_SHADER_2D, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(PARTICLE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            self.particle_shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        
        self.particle_vao = glGenVertexArrays(1)
        glBindVertexArray(self.particle_vao)
        
        self.particle_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
        
        if is_3d:
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        else:
            position_attrib = glGetAttribLocation(self.particle_shader_program, "position")
            glEnableVertexAttribArray(position_attrib)
            glVertexAttribPointer(position_attrib, 2, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        
            type_attrib = glGetAttribLocation(self.particle_shader_program, "particleType")
            glEnableVertexAttribArray(type_attrib)
            glVertexAttribPointer(type_attrib, 1, GL_FLOAT, GL_FALSE, 12 if not is_3d else 16, ctypes.c_void_p(8))
            
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

    def init_surface_rendering(self):
        vertex_shader = shaders.compileShader(SURFACE_VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(SURFACE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.surface_shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        
        self.surface_vao = glGenVertexArrays(1)
        glBindVertexArray(self.surface_vao)

        self.surface_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.surface_vbo)
        
        self.surface_wireframe_vao = glGenVertexArrays(1)
        glBindVertexArray(self.surface_wireframe_vao)

        self.surface_wireframe_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.surface_wireframe_vbo)        

        position_attrib = glGetAttribLocation(self.surface_shader_program, "aPosition")
        glEnableVertexAttribArray(position_attrib)
        glVertexAttribPointer(position_attrib, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)


    def update_particles(self, particle_positions, particle_types, X_max, Y_max, X_min, Y_min):

        assert particle_positions.shape[0] == 2, "Position shape should be (2, n)"
        assert particle_positions.shape[1] == particle_types.shape[0], "Number of positions and types should match"

            
        gl_positions_x = 2.0 * particle_positions[0] - (X_max - X_min)
        gl_positions_y = 2.0 * particle_positions[1] - (Y_max - Y_min)
        gl_positions = cp.array([gl_positions_x, gl_positions_y], dtype=cp.float32)
        gl_positions = cp.ascontiguousarray(gl_positions.T).astype(cp.float32)
        
        particle_data = cp.column_stack((gl_positions, particle_types.astype(cp.float32)))
        particle_data_cpu = particle_data.get()
        
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
        glBufferData(GL_ARRAY_BUFFER, particle_data_cpu.nbytes, particle_data_cpu, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        self.num_particles = gl_positions.shape[0]

    def update_particles_3d(self, particle_positions):
        gl_positions = 2.0 * particle_positions - 1.0
        gl_positions = cp.ascontiguousarray(gl_positions.T).astype(cp.float32)
        particle_positions_cpu = gl_positions.get()
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
        glBufferData(GL_ARRAY_BUFFER, particle_positions_cpu.nbytes, particle_positions_cpu, GL_STATIC_DRAW)
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

    def update_legend(self, label, text, y_pos, color=(0, 255, 0), x_pos=20, scale=0.5):
        self.update_text(label, {
            'text': text,
            'x': x_pos,
            'y': y_pos,
            'scale': scale,
            'color': color
        })

    def clear_legend_entries(self):
        for label in self.text_content:
            self.update_text(label, {
                'text': '',  # Set the text to an empty string
                'x': 0,  # Optionally reset position if needed
                'y': 0,  # Optionally reset position if needed
                'scale': 0.5,
                'color': (0, 0, 0)  # Optionally set the color to black or leave as is
            })

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
        x_normalized = (2.0 * (x_values - cp.min(x_values)) / (cp.max(x_values) - cp.min(x_values)) - 1.0) * 0.9
        y_normalized = (2.0 * (y_values - cp.min(y_values)) / (cp.max(y_values) - cp.min(y_values)) - 1.0) * 0.9

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

    def update_surface(self, X, Y, Z):
        self.z_min = np.min(Z)
        self.z_max = np.max(Z)
        
        def generate_vertices(X, Y, Z):
            assert X.shape == Y.shape == Z.shape, "X, Y, and Z must have the same shape"

            # Generate the 4 corners for each quad
            X1 = X[:-1, :-1]  # Bottom-left
            X2 = X[1:, :-1]   # Bottom-right
            X3 = X[:-1, 1:]   # Top-left
            X4 = X[1:, 1:]    # Top-right

            Y1 = Y[:-1, :-1]
            Y2 = Y[1:, :-1]
            Y3 = Y[:-1, 1:]
            Y4 = Y[1:, 1:]

            Z1 = Z[:-1, :-1]
            Z2 = Z[1:, :-1]
            Z3 = Z[:-1, 1:]
            Z4 = Z[1:, 1:]

            # Create triangles with consistent counter-clockwise winding order
            # First triangle: bottom-left -> top-left -> bottom-right (CCW)
            tri1 = np.dstack([
                X1, Y1, Z1,  # Bottom-left
                X3, Y3, Z3,  # Top-left
                X2, Y2, Z2   # Bottom-right
            ]).reshape(-1, 3)

            # Second triangle: bottom-right -> top-left -> top-right (CCW)
            tri2 = np.dstack([
                X2, Y2, Z2,  # Bottom-right
                X3, Y3, Z3,  # Top-left
                X4, Y4, Z4   # Top-right
            ]).reshape(-1, 3)

            # Combine both triangles
            vertices = np.vstack([tri1, tri2])
            return vertices.astype(np.float32)
        
        def generate_wireframe_vertices(X, Y, Z):
            wireframe_lines = []

            rows, cols = X.shape
            for i in range(rows - 1):
                for j in range(cols - 1):
                    # Horizontal line at top of quad
                    wireframe_lines.append([X[i, j], Y[i, j], Z[i, j]])
                    wireframe_lines.append([X[i, j+1], Y[i, j+1], Z[i, j+1]])
                    
                    # Vertical line on right of quad
                    wireframe_lines.append([X[i, j+1], Y[i, j+1], Z[i, j+1]])
                    wireframe_lines.append([X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]])

            return np.array(wireframe_lines, dtype=np.float32)
        
        vertices = generate_vertices(X, Y, Z)
        self.num_surface_vertices = vertices.shape[0]
        
        # Update OpenGL buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.surface_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        wireframe_vertices = generate_wireframe_vertices(X, Y, Z)
        self.num_wireframe_vertices = wireframe_vertices.shape[0]

        glBindBuffer(GL_ARRAY_BUFFER, self.surface_wireframe_vbo)
        glBufferData(GL_ARRAY_BUFFER, wireframe_vertices.nbytes, wireframe_vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render_particles(self):
        glUseProgram(self.particle_shader_program)
        glBindVertexArray(self.particle_vao)
        glDrawArrays(GL_POINTS, 0, self.num_particles)
        glBindVertexArray(0)

    def render_particles_3d(self, fov=10.0, camera_position=(0.0, 0.0, 1.0), target=(0.0, 0.0, 0.0)):
        """
        Render particles in 3D space with specified camera settings.
        
        :param particle_positions: CuPy array of particle positions with shape (3, n).
        :param fov: Field of view for the perspective projection.
        :param camera_position: Tuple (x, y, z) representing the camera's position.
        :param target: Tuple (x, y, z) representing the point the camera is looking at.
        """
        glUseProgram(self.particle_shader_program)

        # Set up 3D projection matrix
        projection = glm.perspective(glm.radians(fov), self.width / self.height, 0.1, 100.0)
        view = glm.lookAt(glm.vec3(*camera_position), glm.vec3(*target), glm.vec3(0.0, 0.0, 1.0))
        model = glm.mat4(1.0)
        #print(fov, camera_position, target)


        # Pass matrices to the shader
        glUniformMatrix4fv(glGetUniformLocation(self.particle_shader_program, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix4fv(glGetUniformLocation(self.particle_shader_program, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(self.particle_shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))

        # Bind the particle VAO and draw the particles as points
        glBindVertexArray(self.particle_vao)
        glDrawArrays(GL_POINTS, 0, self.num_particles)
        glBindVertexArray(0)

    def render_text(self, text, x, y, scale, color):
        # Use the text shader program
        glUseProgram(self.text_shader_program)
        
        # Set text color
        glUniform3f(glGetUniformLocation(self.text_shader_program, "textColor"),
                    color[0] / 255, color[1] / 255, color[2] / 255)
        
        # Enable blending for transparency
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Bind the VAO
        glBindVertexArray(self.text_vao)
        
        # Prepare vertex buffer for batch rendering
        all_vertices = []
        for c in text:
            ch = self.Characters[c]
            w, h = ch.textureSize
            w *= scale
            h *= scale

            # Compute vertices for this character
            vertices = _get_rendering_buffer(x, y, w, h)
            all_vertices.append(vertices)

            # Advance the cursor position for the next character
            x += (ch.advance >> 6) * scale

        # Flatten the vertices into a single array
        all_vertices = np.vstack(all_vertices).astype(np.float32)

        # Upload all vertices at once
        glBindBuffer(GL_ARRAY_BUFFER, self.text_vbo)
        glBufferData(GL_ARRAY_BUFFER, all_vertices.nbytes, all_vertices, GL_DYNAMIC_DRAW)

        # Render each character
        offset = 0
        for c in text:
            ch = self.Characters[c]

            # Bind the character's texture
            glBindTexture(GL_TEXTURE_2D, ch.texture)

            # Draw the character
            glDrawArrays(GL_TRIANGLES, offset, 6)
            offset += 6

        # Cleanup
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
        
        if len(self.line_plots) == 0:
            print("No line plots to render.")
            return
        elif self.line_plot_type != None:
            
            plot_data = self.line_plots[self.line_plot_type]
            glBindBuffer(GL_ARRAY_BUFFER, plot_data['vbo'])
            position_attrib = glGetAttribLocation(self.line_plot_shader_program, "position")
            glEnableVertexAttribArray(position_attrib)
            glVertexAttribPointer(position_attrib, 2, GL_FLOAT, GL_FALSE, 0, None)

            # Set the color for this line plot
            color_loc = glGetUniformLocation(self.line_plot_shader_program, "lineColor")
            glUniform3f(color_loc, *self.line_colors[self.line_plot_type])

            glDrawArrays(GL_LINE_STRIP, 0, plot_data['num_points'])
            
            glDisableVertexAttribArray(position_attrib)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render_surface(self, z_min, z_max, fov=45.0, cam_pos=(10.0, 10.0, 10.0)):
        glUseProgram(self.surface_shader_program)
        
         # Set up face culling
        #glEnable(GL_CULL_FACE)
        #glCullFace(GL_BACK)
        #glFrontFace(GL_CCW)  # Counter-clockwise winding

        # Set up projection, view, and model matrices
        projection = glm.perspective(glm.radians(fov), 800/600, 0.1, 100.0)
        view = glm.lookAt(glm.vec3(cam_pos), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, 10.0))
        model = glm.mat4(1.0)

        glUniformMatrix4fv(glGetUniformLocation(self.surface_shader_program, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix4fv(glGetUniformLocation(self.surface_shader_program, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(self.surface_shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))

        glUniform1f(glGetUniformLocation(self.surface_shader_program, "zMin"), z_min)
        glUniform1f(glGetUniformLocation(self.surface_shader_program, "zMax"), z_max)

        glBindVertexArray(self.surface_vao)
        glDrawArrays(GL_TRIANGLES, 0, self.num_surface_vertices)
        glBindVertexArray(0)

    def render_surface_wireframe(self, color=(1.0, 1.0, 1.0)):
        glUseProgram(self.surface_shader_program)
        
        # Set uniform color for wireframe
        color_loc = glGetUniformLocation(self.surface_shader_program, "lineColor")
        glUniform3f(color_loc, *color)

        # Bind and render wireframe VAO
        glBindVertexArray(self.surface_wireframe_vao)
        glDrawArrays(GL_LINES, 0, self.num_wireframe_vertices)
        glBindVertexArray(0)

    def render(self, clear=True, TEXT_RENDERING=True):
        if clear:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background

        if self.renderer_type == "particles":
            if self.is_3d:
                glEnable(GL_DEPTH_TEST)
                # Set projection and view matrices for 3D
                self.render_particles_3d(fov=self.fov, camera_position=self.eye, target=(0.0, 0.0, 0.0))
            self.render_particles()
        elif self.renderer_type == "heatmap":
            self.render_heatmap()
        elif self.renderer_type == "line_plot":
            self.render_line_plots()
        elif self.renderer_type == "surface_plot":
            self.render_surface(self.z_min, self.z_max, self.fov, self.eye)
            self.render_surface_wireframe(color=(0.0, 0.0, 0.0))
        # Render all text entries
        if TEXT_RENDERING:
            for label in self.label_list:
                content = self.text_content[label]
                self.render_text(content['text'], content['x'], content['y'], content['scale'], content['color'])

        glfw.swap_buffers(self.window)
        glfw.poll_events()
    
    def should_close(self):
        return glfw.window_should_close(self.window)

    def close(self):
        glfw.terminate()


class Simple3DParticleRenderer:
    def __init__(self, width=800, height=600, use_orthographic=False):

        self.width = width
        self.height = height
        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW can't be initialized")
        
        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(width, height, "3D Particle Renderer", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window can't be created")
        
        # Make the window's context current
        glfw.make_context_current(self.window)
        glEnable(GL_DEPTH_TEST)  # Enable depth testing for 3D

        # Compile shaders
        self.shader_program = self.load_shader()
        
        # Set up default camera parameters
        self.use_orthographic = use_orthographic
        self.fov = 45.0  # Only used if `use_orthographic` is False
        self.angle_phi = 0.0
        self.angle_theta = 0.0
        self.distance = 10.0

        # Orthographic projection parameters
        self.left = -0.5
        self.right = 0.5
        self.bottom = -0.5
        self.top = 0.5
        self.near = 0.1
        self.far = 100.0

        # Initialize buffer objects
        self.particle_VBO = None
        self.num_particles = 0  # Number of particles to render

    def load_shader(self):
        vertex_shader_src = """
        #version 330 core
        layout (location = 0) in vec3 position;
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;
        void main() {
            vec4 worldPosition = model * vec4(position, 1.0);
            gl_Position = projection * view * worldPosition;
            gl_PointSize = 5.0;
        }
        """
        fragment_shader_src = """
        #version 330 core
        out vec4 fragColor;
        void main() {
            fragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
        """
        # Compile and link shaders
        vertex_shader = shaders.compileShader(vertex_shader_src, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(fragment_shader_src, GL_FRAGMENT_SHADER)
        shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        return shader_program

    def setup_particles(self, particle_positions):
        # Flatten the particle positions and convert to float32
        particle_vertices = particle_positions.flatten().astype(np.float32)
        self.num_particles = particle_positions.shape[0]  # Store number of particles
        
        # Generate and bind buffer
        self.particle_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_VBO)
        glBufferData(GL_ARRAY_BUFFER, particle_vertices.nbytes, particle_vertices, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    
    def render(self):
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up projection matrix (orthographic or perspective)
        if self.use_orthographic:
            projection = glm.ortho(-self.distance, self.distance, -self.distance, self.distance, self.near, self.far)
        else:
            projection = glm.perspective(glm.radians(self.fov), self.width / self.height, 0.1, 100.0)
        
        # Set up view matrix
        view = glm.lookAt(glm.vec3(self.distance * np.cos(self.angle_phi), 
                                   self.distance * np.sin(self.angle_phi), 
                                   self.distance * np.sin(self.angle_theta)), 
                          glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, 1.0))
        model = glm.mat4(1.0)

        # Use the shader program and set matrices
        glUseProgram(self.shader_program)
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        view_loc = glGetUniformLocation(self.shader_program, "view")
        model_loc = glGetUniformLocation(self.shader_program, "model")
        
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
        
        # Render particles
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_VBO)
        glDrawArrays(GL_POINTS, 0, self.num_particles)  # Use self.num_particles
        
        # Swap buffers
        glfw.swap_buffers(self.window)
    
    def update_camera(self, fov=None, angle_phi=None, angle_theta=None, distance=None):
        # Update camera parameters
        if fov is not None: self.fov = fov
        if angle_phi is not None: self.angle_phi = angle_phi
        if angle_theta is not None: self.angle_theta = angle_theta
        if distance is not None: self.distance = distance

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