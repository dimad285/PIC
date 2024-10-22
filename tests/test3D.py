import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm

def initialize_window(width=800, height=600):
    # Initialize GLFW
    if not glfw.init():
        raise Exception("GLFW can't be initialized")

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(800, 600, "3D Surface Plot", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window can't be created")

    # Make the window's context current
    glfw.make_context_current(window)
    return window


def generate_surface_data():
    # Generate a grid of points (X, Y) and corresponding Z values for the surface
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))  # Example surface (circular wave)
    return X, Y, Z


def generate_vertices(X, Y, Z):
    vertices = []
    for i in range(len(X) - 1):
        for j in range(len(Y) - 1):
            # Create two triangles per grid square
            v1 = (X[i, j], Y[i, j], Z[i, j])
            v2 = (X[i+1, j], Y[i+1, j], Z[i+1, j])
            v3 = (X[i, j+1], Y[i, j+1], Z[i, j+1])
            v4 = (X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1])
            
            # First triangle
            vertices.extend(v1)
            vertices.extend(v2)
            vertices.extend(v3)

            # Second triangle
            vertices.extend(v2)
            vertices.extend(v4)
            vertices.extend(v3)
    
    return np.array(vertices, dtype=np.float32)



def setup_opengl(vertices):
    # Create a Vertex Buffer Object (VBO) to store the vertex data on the GPU
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Define the structure of the vertex data
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)


def load_shader(shader_type):
    """
    Load and compile vertex and fragment shaders based on the shader type ('surface' or 'particle').
    """

    if shader_type == 'surface':
        # Surface shaders as strings
        vertex_shader_src = """
        #version 330 core

        layout (location = 0) in vec3 position;

        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;

        out float vertexZ;

        void main() {
            vec4 worldPosition = model * vec4(position, 1.0);
            vertexZ = worldPosition.z;
            gl_Position = projection * view * worldPosition;
        }
        """
        fragment_shader_src = """
        #version 330 core

        in float vertexZ;
        out vec4 fragColor;

        uniform float zMin;
        uniform float zMax;

        void main() {
            float normalizedZ = (vertexZ - zMin) / (zMax - zMin);
            vec3 lowColor = vec3(0.0, 0.0, 1.0);  // Blue for low Z
            vec3 highColor = vec3(1.0, 0.0, 0.0);  // Red for high Z
            vec3 finalColor = mix(lowColor, highColor, normalizedZ);  // Gradient color
            fragColor = vec4(finalColor, 1.0);  // Output color
        }
        """

    elif shader_type == 'particle':
        # Particle shaders as strings
        vertex_shader_src = """
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
        fragment_shader_src = """
        #version 330 core

        out vec4 fragColor;

        void main() {
            fragColor = vec4(1.0, 1.0, 1.0, 1.0);  // White color for particles
        }
        """

    else:
        raise ValueError("Unknown shader type. Use 'surface' or 'particle'.")

    # Compile vertex shader
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_src)
    glCompileShader(vertex_shader)
    if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(vertex_shader))

    # Compile fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_src)
    glCompileShader(fragment_shader)
    if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(fragment_shader))

    # Link shaders into a program
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(shader_program))

    # Cleanup shaders after linking
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program

def render_loop_with_separate_shaders(window, surface_vertices, particle_vertices, surface_shader_program, particle_shader_program, z_min, z_max, render_type='surface'):
    # Enable depth testing for 3D rendering
    glEnable(GL_DEPTH_TEST)

    # Set up projection and view matrices
    projection = glm.perspective(glm.radians(45.0), 800 / 600, 0.1, 100.0)
    view = glm.lookAt(glm.vec3(0.0, 0.0, 10.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
    model = glm.mat4(1.0)

    while not glfw.window_should_close(window):
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Render the surface
        if render_type == 'surface':
            render_surface(surface_vertices, surface_shader_program, z_min, z_max, projection, view, model)

        elif render_type == 'particle':
            render_particles(particle_vertices, particle_shader_program, projection, view, model)

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


def render_surface(surface_vertices, surface_shader_program, z_min, z_max, proj_matrix, view_matrix, model_matrix):
    # Bind the surface VBO
    glBindBuffer(GL_ARRAY_BUFFER, surface_VBO)

    # Use the surface shader program
    glUseProgram(surface_shader_program)

    # Set up matrices
    proj_loc = glGetUniformLocation(surface_shader_program, "projection")
    view_loc = glGetUniformLocation(surface_shader_program, "view")
    model_loc = glGetUniformLocation(surface_shader_program, "model")
    z_min_loc = glGetUniformLocation(surface_shader_program, "zMin")
    z_max_loc = glGetUniformLocation(surface_shader_program, "zMax")

    # Set uniform values
    glUniform1f(z_min_loc, z_min)
    glUniform1f(z_max_loc, z_max)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(proj_matrix))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view_matrix))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model_matrix))

    # Render the surface
    glDrawArrays(GL_TRIANGLES, 0, len(surface_vertices) // 3)  # Render surface

def render_particles(particle_vertices, particle_shader_program, proj_matrix, view_matrix, model_matrix):
    # Bind the particle VBO
    glBindBuffer(GL_ARRAY_BUFFER, particle_VBO)

    # Use the particle shader program
    glUseProgram(particle_shader_program)

    # Set up matrices
    proj_loc = glGetUniformLocation(particle_shader_program, "projection")
    view_loc = glGetUniformLocation(particle_shader_program, "view")
    model_loc = glGetUniformLocation(particle_shader_program, "model")

    # Set uniform values
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(proj_matrix))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view_matrix))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model_matrix))

    # Render the particles
    glDrawArrays(GL_POINTS, 0, len(particle_vertices) // 3)  # Render particles

def generate_particle_data(particles):
    """
    Convert particle positions from a (3, n) array to a 1D array of vertices.
    """
    return particles.flatten().astype(np.float32)


def setup_opengl_surface(surface_vertices):
    """
    Set up OpenGL buffers for the surface rendering.
    """
    surface_VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, surface_VBO)
    glBufferData(GL_ARRAY_BUFFER, surface_vertices.nbytes, surface_vertices, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    return surface_VBO

def setup_opengl_particles(particle_vertices):
    """
    Set up OpenGL buffers for the particle rendering.
    """
    particle_VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, particle_VBO)
    glBufferData(GL_ARRAY_BUFFER, particle_vertices.nbytes, particle_vertices, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    return particle_VBO

window = initialize_window()

X, Y, Z = generate_surface_data()

z_min = np.min(Z)
z_max = np.max(Z)

surface_vertices = generate_vertices(X, Y, Z)

# Set up OpenGL buffers and shaders
#setup_opengl(vertices)
shader_program_surface = load_shader('surface')
shader_program_particle = load_shader('particle')

# Enter the rendering loop, passing z_min and z_max

# Example particle data: Random particles in 3D space
particle_data = np.random.rand(3, 100) * 10 - 5  # 100 random particles in the range [-5, 5]
particle_vertices = generate_particle_data(particle_data)

# Set up OpenGL buffers and shaders for both surface and particles
#setup_opengl(vertices)  # Surface
surface_VBO = setup_opengl_surface(surface_vertices)
particle_VBO = setup_opengl_particles(particle_vertices)  # Particles

# Enter the rendering loop with particles
render_loop_with_separate_shaders(window, surface_vertices, particle_vertices, shader_program_surface, shader_program_particle, z_min, z_max, render_type='surface')
