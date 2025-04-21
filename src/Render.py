import numpy as np
import cupy as cp
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import ctypes
import freetype
import glm
from cuda import cudart
from CuPy_TO_OpenGL import *
from dataclasses import dataclass, field
import numpy as np
from math import radians, cos, sin
import Particles

@dataclass
class Camera:
    fov: float = 45  # Field of view in degrees
    r: float = 1.0  # Distance from the origin
    phi: float = 0.0  # Azimuthal angle in radians
    theta: float = 0.0  # Polar angle in radians
    x: float = 0.0  # Offset along the X-axis
    y: float = 0.0  # Offset along the Y-axis
    z: float = 0.0  # Offset along the Z-axis
    
    @property
    def fov_radians(self) -> float:
        """Convert field of view to radians."""
        return radians(self.fov)

    @property
    def position(self) -> np.ndarray:
        """Compute the camera position in 3D space."""
        return np.array([
            self.r * cos(self.phi) + self.x,
            self.r * sin(self.phi) + self.y,
            self.r * sin(self.theta) + self.z
        ])

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
    def __init__(self, width, height, fontfile, max_particles, renderer_type="particles", is_3d=False):
        self.width = width
        self.height = height
        self.fontfile = fontfile
        self.renderer_type = renderer_type  # Choose either 'particles' or 'heatmap'
        self.text_content = {}
        self.is_3d = is_3d
        self.line_plot_type = None
        self.fov = np.radians(60)
        self.label_list = []
        
        # Remove downsampling configuration
        self.particle_count = max_particles
        
        print(f'Initializing {renderer_type} PIC Renderer...')

        # Rest of initialization
        self.use_orthographic = False
        self.angle_phi = 0.0
        self.angle_theta = 0.0
        self.distance = 10.0
        self.left = -0.5
        self.right = 0.5
        self.bottom = -0.5
        self.top = 0.5
        self.near = 0.1
        self.far = 100.0
        self.eye = np.array([0.0, 0.0, 10.0])
        
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        self.window = glfw.create_window(width, height, "Multi-type PIC Renderer with Text (CuPy)", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.set_window_size_callback(self.window, self._handle_window_resize)
        glfw.make_context_current(self.window)

        # Initialize all rendering components
        self.init_particle_rendering()
        self.init_heatmap_rendering()
        self.init_line_plot_rendering()
        self.init_surface_rendering()
        self.init_text_rendering()
        self.init_boundary_rendering()

        glfw.swap_interval(0)

        self.line_plots = {}  # Dictionary to store multiple line plots
        self.line_colors = {}  # Dictionary to store colors for each line plot
        self.num_line_points = 0  # Initialize num_line_pointss

        self.last_mouse_pos = None
        self.mouse_buttons = {'left': False, 'right': False, 'middle': False}
        
        # Set up input callbacks
        glfw.set_cursor_pos_callback(self.window, self._handle_mouse_move)
        glfw.set_mouse_button_callback(self.window, self._handle_mouse_button)
        glfw.set_scroll_callback(self.window, self._handle_scroll)
        glfw.set_key_callback(self.window, self._handle_keyboard)

    def set_title(self, title):
        glfw.set_window_title(self.window, title)
    
    def set_renderer_type(self, renderer_type, is_3d=False):
        """Set the renderer type and whether it's 2D or 3D."""
        self.renderer_type = renderer_type
        self.is_3d = is_3d  # Boolean to control 2D or 3D mode
        self.init_particle_rendering(is_3d=is_3d)

    def set_fov(self, fov):
        self.fov = fov

    def set_camera(self, eye):
        self.eye = eye

    def init_particle_rendering(self):


        PARTICLE_VERTEX_SHADER = """
        #version 330
        in vec3 position;
        in float type;
        out float vParticleType;  
        uniform mat4 projection;
        uniform mat4 modelView;
        void main() {
            gl_Position = projection * modelview * vec4(position, 1.0);
            vParticleType = type;
        }
        """ if self.is_3d else """
        #version 330
        in vec2 position;
        in float type;  
        out float vParticleType;  
        uniform mat4 projection;
        uniform mat4 modelview;
        void main() {
            gl_Position = projection * modelview * vec4(position, 0.0, 1.0);
            vParticleType = type; 
        }
        """

        PARTICLE_FRAGMENT_SHADER = """
        #version 330
        in float vParticleType; 
        out vec4 fragColor;

        void main() {
            if (vParticleType == 0.0) fragColor = vec4(0.0, 0.0, 0.0, 1.0);  
            else if (vParticleType == 1.0) fragColor = vec4(0.0, 1.0, 0.0, 1.0);  
            else if (vParticleType == 2.0) fragColor = vec4(1.0, 0.5, 0.0, 1.0);  
            else if (vParticleType == 3.0) fragColor = vec4(1.0, 1.0, 0.0, 1.0);  
            else fragColor = vec4(1.0, 1.0, 1.0, 1.0);  
        }
        """
        # Compile shaders
        vertex_shader = shaders.compileShader(PARTICLE_VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(PARTICLE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.particle_shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        
        # Get uniform locations
        self.particle_mvp_loc = glGetUniformLocation(self.particle_shader_program, "MVP")
        self.projection_loc = glGetUniformLocation(self.particle_shader_program, "projection")
        self.modelview_loc = glGetUniformLocation(self.particle_shader_program, "modelview")
        
        # Setup vertex attributes
        particle_vertex_size = 4 if self.is_3d else 3  # [x, y, type] or [x, y, z, type]
        self.particle_vertex_size = particle_vertex_size
        
        # Allocate buffer for max_displayed_particles
        vertex_bytes = self.particle_count * particle_vertex_size * np.float32().nbytes
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
        
        self.particle_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_bytes, None, GL_DYNAMIC_DRAW)
        
        self.particle_buffer = CudaOpenGLMappedArray(
            np.float32, 
            (self.particle_count, particle_vertex_size), 
            self.particle_vbo, 
            flags
        )
        
        # Setup vertex attributes
        glEnableVertexAttribArray(0)  # position
        glEnableVertexAttribArray(1)  # type
        if self.is_3d:
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, particle_vertex_size * 4, ctypes.c_void_p(0))
            glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, particle_vertex_size * 4, ctypes.c_void_p(12))
        else:
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, particle_vertex_size * 4, ctypes.c_void_p(0))
            glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, particle_vertex_size * 4, ctypes.c_void_p(8))

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

    def init_boundary_rendering(self):
        vertex_shader = shaders.compileShader(LINE_PLOT_VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(LINE_PLOT_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.boundary_shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        
        self.boundary_vao = glGenVertexArrays(1)
        glBindVertexArray(self.boundary_vao)
        
        self.boundary_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.boundary_vbo)
        
        position_attrib = glGetAttribLocation(self.boundary_shader_program, "position")
        glEnableVertexAttribArray(position_attrib)
        glVertexAttribPointer(position_attrib, 2, GL_FLOAT, GL_FALSE, 0, None)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def update_particles(self, particles: Particles.Particles2D, x_min, x_max, y_min, y_max):
        """
        Update particle positions.
        
        :param particles: Particles2D object containing particle data
        :param x_min, x_max, y_min, y_max: Domain boundaries
        """
        num_particles = particles.last_alive
        particle_positions = particles.R[:num_particles]
        particle_types = particles.part_type[:num_particles]

        assert num_particles <= self.particle_count, (
            f"Number of particles ({num_particles}) exceeds buffer size ({self.particle_count})."
        )
        assert particle_positions.shape[1] == 2, "particle_positions must be of shape (n, 2)"

        # Normalize particle positions to OpenGL space [-1, 1]
        normalized_positions = cp.empty_like(particle_positions)
        normalized_positions[:, 0] = 2.0 * (particle_positions[:, 0] - x_min) / (x_max - x_min) - 1.0
        normalized_positions[:, 1] = 2.0 * (particle_positions[:, 1] - y_min) / (y_max - y_min) - 1.0

        # Create interleaved data [x, y, type]
        interleaved_data = cp.empty((num_particles, self.particle_vertex_size), dtype=cp.float32)
        interleaved_data[:, :2] = normalized_positions
        interleaved_data[:, 2] = particle_types.astype(cp.float32)

        with self.particle_buffer as particle_data:
            particle_data[:num_particles] = interleaved_data

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
        # Reshape the CuPy 1D data to 2D (rows, cols)
        heatmap_data_cupy = data_1d_cupy.reshape((rows, cols))
        
        # Normalize intensity values between 0 and 1
        heatmap_min = cp.min(heatmap_data_cupy)
        heatmap_max = cp.max(heatmap_data_cupy)
        # Avoid division by zero if all values are the same
        if heatmap_max == heatmap_min:
            heatmap_data_cupy_normalized = cp.zeros_like(heatmap_data_cupy)
        else:
            heatmap_data_cupy_normalized = (heatmap_data_cupy - heatmap_min) / (heatmap_max - heatmap_min)
        
        # Create vertex data for quads using CuPy operations
        # Use proper dimensions for vertex layout
        x = cp.linspace(-1, 1, cols)
        y = cp.linspace(-1, 1, rows)
        X, Y = cp.meshgrid(x, y)  # This gives X with shape (rows, cols)
        
        # Create vertices array with correct dimensions (rows-1, cols-1, 6, 3)
        vertices = cp.zeros((rows-1, cols-1, 6, 3), dtype=cp.float32)
        
        # First triangle (bottom-left, bottom-right, top-left)
        vertices[:,:,0,0] = X[:-1,:-1]  # Bottom-left x
        vertices[:,:,0,1] = Y[:-1,:-1]  # Bottom-left y
        vertices[:,:,0,2] = heatmap_data_cupy_normalized[:-1,:-1]  # Bottom-left intensity
        
        vertices[:,:,1,0] = X[:-1,1:]   # Bottom-right x
        vertices[:,:,1,1] = Y[:-1,1:]   # Bottom-right y
        vertices[:,:,1,2] = heatmap_data_cupy_normalized[:-1,1:]  # Bottom-right intensity
        
        vertices[:,:,2,0] = X[1:,:-1]   # Top-left x
        vertices[:,:,2,1] = Y[1:,:-1]   # Top-left y
        vertices[:,:,2,2] = heatmap_data_cupy_normalized[1:,:-1]  # Top-left intensity
        
        # Second triangle (top-left, bottom-right, top-right)
        vertices[:,:,3,0] = X[1:,:-1]   # Top-left x
        vertices[:,:,3,1] = Y[1:,:-1]   # Top-left y
        vertices[:,:,3,2] = heatmap_data_cupy_normalized[1:,:-1]  # Top-left intensity
        
        vertices[:,:,4,0] = X[:-1,1:]   # Bottom-right x
        vertices[:,:,4,1] = Y[:-1,1:]   # Bottom-right y
        vertices[:,:,4,2] = heatmap_data_cupy_normalized[:-1,1:]  # Bottom-right intensity
        
        vertices[:,:,5,0] = X[1:,1:]    # Top-right x
        vertices[:,:,5,1] = Y[1:,1:]    # Top-right y
        vertices[:,:,5,2] = heatmap_data_cupy_normalized[1:,1:]   # Top-right intensity
        
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
            rows, cols = X.shape
            
            # Pre-allocate arrays for horizontal and vertical lines
            # For horizontal lines
            h_indices_row = np.repeat(np.arange(rows), cols-1)
            h_indices_col_start = np.tile(np.arange(cols-1), rows)
            h_indices_col_end = h_indices_col_start + 1
            
            # For vertical lines
            v_indices_col = np.repeat(np.arange(cols), rows-1)
            v_indices_row_start = np.tile(np.arange(rows-1), cols)
            v_indices_row_end = v_indices_row_start + 1
            
            # Extract start and end points for horizontal lines
            h_start_x = X[h_indices_row, h_indices_col_start]
            h_start_y = Y[h_indices_row, h_indices_col_start]
            h_start_z = Z[h_indices_row, h_indices_col_start]
            h_end_x = X[h_indices_row, h_indices_col_end]
            h_end_y = Y[h_indices_row, h_indices_col_end]
            h_end_z = Z[h_indices_row, h_indices_col_end]
            
            # Extract start and end points for vertical lines
            v_start_x = X[v_indices_row_start, v_indices_col]
            v_start_y = Y[v_indices_row_start, v_indices_col]
            v_start_z = Z[v_indices_row_start, v_indices_col]
            v_end_x = X[v_indices_row_end, v_indices_col]
            v_end_y = Y[v_indices_row_end, v_indices_col]
            v_end_z = Z[v_indices_row_end, v_indices_col]
            
            # Combine horizontal and vertical line start/end points
            start_points = np.column_stack([
                np.hstack([h_start_x, v_start_x]),
                np.hstack([h_start_y, v_start_y]),
                np.hstack([h_start_z, v_start_z])
            ])
            
            end_points = np.column_stack([
                np.hstack([h_end_x, v_end_x]),
                np.hstack([h_end_y, v_end_y]),
                np.hstack([h_end_z, v_end_z])
            ])
            
            # Create the final wireframe vertices
            wireframe_lines = np.hstack([start_points, end_points]).astype(np.float32)
            return wireframe_lines
        
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

    def update_boundaries(self, boundary_segments, grid_shape):
        """
        Update the boundary data using a list of line segments.
        
        :param boundary_segments: List of tuples [(x0, y0, x1, y1), ...] defining the line segments.
        :param grid_shape: Tuple (rows, cols) defining the dimensions of the grid.
        """
        rows, cols = grid_shape
        
        # Convert to normalized OpenGL coordinates
        normalized_segments = []
        for x0, y0, x1, y1 in boundary_segments:
            x0_normalized = 2.0 * x0 / (cols-1) - 1.0
            y0_normalized = 2.0 * y0 / (rows-1) - 1.0
            x1_normalized = 2.0 * x1 / (cols-1) - 1.0
            y1_normalized = 2.0 * y1 / (rows-1) - 1.0
            normalized_segments.extend([(x0_normalized, y0_normalized), (x1_normalized, y1_normalized)])
        
        # Convert to NumPy array
        boundary_lines = np.array(normalized_segments, dtype=np.float32)
        
        # Update OpenGL buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.boundary_vbo)
        glBufferData(GL_ARRAY_BUFFER, boundary_lines.nbytes, boundary_lines, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        self.num_boundary_vertices = len(boundary_lines)

    def render_particles(self, projection_matrix=None, modelview_matrix=None):
        """
        Render particles using the updated buffer.
        """
        glUseProgram(self.particle_shader_program)

        if projection_matrix is None:
            projection_matrix = np.eye(4, dtype=np.float32)
        if modelview_matrix is None:
            modelview_matrix = np.eye(4, dtype=np.float32)

        # Pass projection and modelView matrices to the shader
        glUniformMatrix4fv(self.projection_loc, 1, GL_FALSE, projection_matrix)
        glUniformMatrix4fv(self.modelview_loc, 1, GL_FALSE, modelview_matrix)
        
        # Bind particle buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
    
        # Position attribute
        position_attrib = glGetAttribLocation(self.particle_shader_program, "position")
        glEnableVertexAttribArray(position_attrib)
        stride = (self.particle_vertex_size) * np.float32().nbytes
        glVertexAttribPointer(position_attrib, self.particle_vertex_size-1, GL_FLOAT, GL_FALSE, stride, None)
        
        # Type attribute
        type_attrib = glGetAttribLocation(self.particle_shader_program, "type")
        glEnableVertexAttribArray(type_attrib)
        offset = (self.particle_vertex_size-1) * np.float32().nbytes
        glVertexAttribPointer(type_attrib, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        
        glDrawArrays(GL_POINTS, 0, self.particle_count)
        
        glDisableVertexAttribArray(position_attrib)
        glDisableVertexAttribArray(type_attrib)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

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
        linked = glGetProgramiv(self.text_shader_program, GL_LINK_STATUS)
        if not linked:
            log = glGetProgramInfoLog(self.text_shader_program)
            print("Shader program failed to link:\n", log.decode())
        # Set text color (normalize to 0.0 - 1.0)
        glUniform3f(glGetUniformLocation(self.text_shader_program, "textColor"),
                    color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

        # Enable blending for transparency
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Bind the VAO
        glBindVertexArray(self.text_vao)

        # Prepare vertex buffer for batch rendering
        
        all_vertices = []
        for c in text:
            ch = self.Characters.get(c)
            if ch is None:
                continue  # Skip unknown characters

            w, h = ch.textureSize
            w *= scale
            h *= scale

            # Compute vertices for this character
            vertices = _get_rendering_buffer(x + ch.bearing[0] * scale, 
                                            y - (ch.textureSize[1] - ch.bearing[1]) * scale, 
                                            w, h)
            all_vertices.append(vertices)

            # Advance the cursor position for the next character
            x += (ch.advance >> 6) * scale  # Bitshift by 6 to get pixel value (1/64th of a pixel)


        if not all_vertices:
            return  # Nothing to render

        # Flatten the vertices into a single array
        all_vertices = np.vstack(all_vertices).astype(np.float32)

        # Upload all vertices at once
        glBindBuffer(GL_ARRAY_BUFFER, self.text_vbo)
        glBufferData(GL_ARRAY_BUFFER, all_vertices.nbytes, all_vertices, GL_DYNAMIC_DRAW)

        # Render each character
        offset = 0
        for c in text:
            ch = self.Characters.get(c)
            if ch is None:
                continue

            glBindTexture(GL_TEXTURE_2D, ch.texture)
            glDrawArrays(GL_TRIANGLES, offset, 6)
            offset += 6

        # Cleanup
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_BLEND)



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

    def render_boundaries(self, color=(1.0, 0.0, 0.0)):
        glUseProgram(self.boundary_shader_program)
        glUniform3f(glGetUniformLocation(self.boundary_shader_program, "lineColor"), *color)
        
        glBindVertexArray(self.boundary_vao)
        glDrawArrays(GL_LINES, 0, self.num_boundary_vertices)
        glBindVertexArray(0)

    def render(self, clear=True, TEXT_RENDERING=True, GRID_RENDERING=False, BOUNDARY_RENDERING=False):
        if clear:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background

        if self.renderer_type == "particles":
            if self.is_3d:
                glEnable(GL_DEPTH_TEST)
                # Set projection and view matrices for 3D
                self.render_particles_3d(fov=self.fov, camera_position=self.eye, target=(0.0, 0.0, 0.0))
            else:
                self.render_particles()
            if GRID_RENDERING:
                pass
            if BOUNDARY_RENDERING:
                self.render_boundaries()
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
        """Properly cleanup CUDA and OpenGL resources"""
        if self.particle_buffer is not None:
            self.particle_buffer.unregister()
            self.particle_buffer = None
        
        if hasattr(self, 'window') and self.window is not None:
            glfw.make_context_current(self.window)
            if hasattr(self, 'particle_vbo'):
                glDeleteBuffers(1, [self.particle_vbo])
            glfw.destroy_window(self.window)
            self.window = None
        
        glfw.terminate()

    def __del__(self):
        """Ensure resources are cleaned up if object is garbage collected"""
        self.close()

    def _handle_mouse_move(self, window, xpos, ypos):
        if self.last_mouse_pos is None:
            self.last_mouse_pos = (xpos, ypos)
            return
            
        dx = xpos - self.last_mouse_pos[0]
        dy = ypos - self.last_mouse_pos[1]
        
        # Rotate camera with left mouse button
        if self.mouse_buttons['left']:
            self.angle_phi -= dx * 0.005
            self.angle_theta = max(min(self.angle_theta + dy * 0.005, np.pi/2), -np.pi/2)
            
        # Pan camera with right mouse button
        if self.mouse_buttons['right']:
            self.left += dx * 0.01
            self.top += dy * 0.01
            self.right += dx * 0.01
            self.bottom += dy * 0.01
            
        self.last_mouse_pos = (xpos, ypos)

    def _handle_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_buttons['left'] = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_buttons['right'] = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.mouse_buttons['middle'] = action == glfw.PRESS
            
        if action == glfw.RELEASE:
            self.last_mouse_pos = None

    def _handle_scroll(self, window, xoffset, yoffset):
        # Zoom with mouse wheel
        zoom_factor = 0.1
        if self.use_orthographic:
            scale = 1.0 - yoffset * zoom_factor
            self.left *= scale
            self.right *= scale
            self.top *= scale
            self.bottom *= scale
        else:
            self.distance = max(1.0, self.distance * (1.0 - yoffset * zoom_factor))

    def _handle_keyboard(self, window, key, scancode, action, mods):
        if action != glfw.PRESS and action != glfw.REPEAT:
            return
            
        if key == glfw.KEY_R:  # Reset view
            self.reset_camera()
        elif key == glfw.KEY_P:  # Toggle projection mode
            self.use_orthographic = not self.use_orthographic
        elif key == glfw.KEY_UP:
            self.fov = max(10.0, self.fov - 5.0)
        elif key == glfw.KEY_DOWN:
            self.fov = min(120.0, self.fov + 5.0)

    def reset_camera(self):
        """Reset camera to default parameters"""
        self.angle_phi = 0.0
        self.angle_theta = 0.0
        self.distance = 10.0
        self.fov = 45.0
        self.left = -0.5
        self.right = 0.5
        self.bottom = -0.5
        self.top = 0.5

    def resize_particle_buffer(self, new_max_particles):
        """Resize the OpenGL/CUDA buffer when max_displayed_particles changes significantly"""
        # Delete old buffer
        glDeleteBuffers(1, [self.particle_vbo])
        self.particle_buffer = None

        # Create new buffer
        vertex_bytes = new_max_particles * self.particle_vertex_size * np.float32().nbytes
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard

        self.particle_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_bytes, None, GL_DYNAMIC_DRAW)
        
        self.particle_buffer = CudaOpenGLMappedArray(
            np.float32, 
            (new_max_particles, self.particle_vertex_size), 
            self.particle_vbo, 
            flags
        )
        
        self.max_displayed_particles = new_max_particles

    def adjust_particle_count(self, current_frame_time):
        """Dynamically adjust max_displayed_particles based on frame time"""
        if not self.adaptive_sampling:
            return
            
        old_max = self.max_displayed_particles
        
        # ... existing adjustment code ...
        
        # If max_displayed_particles changed significantly, resize buffer
        if abs(old_max - self.max_displayed_particles) / old_max > 0.5:  # 50% change
            self.resize_particle_buffer(self.max_displayed_particles)


    def _handle_window_resize(self, window, width, height):
        """Handle window resize events by updating viewport to match window dimensions."""
        glViewport(0, 0, width, height)
        self.width = width
        self.height = height
