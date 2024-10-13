import numpy as np
import cupy as cp
import sdl2
import sdl2.ext
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import ctypes


# Vertex shader
VERTEX_SHADER = """
#version 330
in vec2 position;
in float particleType;
out float vParticleType;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    vParticleType = particleType;
}
"""

# Fragment shader
FRAGMENT_SHADER = """
#version 330
in float vParticleType;
out vec4 fragColor;

vec3 getColor(float type) {
    if (type == 0.0) return vec3(1.0, 0.0, 0.0);  // Red
    else if (type == 1.0) return vec3(0.0, 1.0, 0.0);  // Green
    else if (type == 2.0) return vec3(1.0, 0.5, 0.0);  // Orange
    else if (type == 3.0) return vec3(1.0, 1.0, 0.0);  // Yellow
    else return vec3(1.0, 1.0, 1.0);  // White (default)
}

void main() {
    fragColor = vec4(getColor(vParticleType), 1.0);
}
"""

class PICRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        self.window = glfw.create_window(width, height, "Multi-type PIC Renderer (CuPy)", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.make_context_current(self.window)

        vertex_shader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        
        position_attrib = glGetAttribLocation(self.shader_program, "position")
        glEnableVertexAttribArray(position_attrib)
        glVertexAttribPointer(position_attrib, 2, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        
        type_attrib = glGetAttribLocation(self.shader_program, "particleType")
        glEnableVertexAttribArray(type_attrib)
        glVertexAttribPointer(type_attrib, 1, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(8))
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glfw.swap_interval(0)

    def update_particles(self, particle_positions, particle_types):
        """
        Update particle positions and types.
        particle_positions: CuPy array of shape (2, n) with values in the range [0, 1]
        particle_types: CuPy array of shape (n,) with integer values
        """
        assert particle_positions.shape[0] == 2, "Position shape should be (2, n)"
        assert particle_positions.shape[1] == particle_types.shape[0], "Number of positions and types should match"
        
        gl_positions = 2.0 * particle_positions - 1.0
        gl_positions = cp.ascontiguousarray(gl_positions.T).astype(cp.float32)
        
        # Combine positions and types into a single array
        particle_data = cp.column_stack((gl_positions, particle_types.astype(cp.float32)))
        
        # Transfer data from GPU to CPU
        particle_data_cpu = particle_data.get()
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, particle_data_cpu.nbytes, particle_data_cpu, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        self.num_particles = gl_positions.shape[0]

    def render(self):
        glfw.poll_events()
        
        glClear(GL_COLOR_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background
        
        glUseProgram(self.shader_program)
        
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.num_particles)
        glBindVertexArray(0)
        
        glfw.swap_buffers(self.window)

    def should_close(self):
        return glfw.window_should_close(self.window)
    
    def change_title(self, title):
        glfw.set_window_title(self.window, title)

    def close(self):
        glfw.terminate()



# Example usage with simulated PIC data
if __name__ == "__main__":
    width, height = 800, 600
    renderer = PICRenderer(width, height)
    
    n = 100
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    
    t = 0
    while not renderer.should_close():
        # Simulate changing PIC data
        Z = np.sin(2 * np.pi * (X + Y + t))
        surface_data = np.dstack((X, Y, Z))
        
        renderer.update_surface(surface_data)
        renderer.render()
        
        t += 0.05
    
    renderer.close()