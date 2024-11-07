import tkinter as tk
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

def load_shader():
    """
    Load and compile vertex and fragment shaders based on the shader type ('surface' or 'particle').
    """

    
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

def render(window, particle_vertices, particle_shader_program, fov, angle_phi, angle_theta, distance):
    # Enable depth testing for 3D rendering
    glEnable(GL_DEPTH_TEST)

    # Set up projection and view matrices
    projection = glm.perspective(glm.radians(fov), 800 / 600, 0.1, 100.0)
    view = glm.lookAt(glm.vec3(distance * np.cos(angle_phi), distance * np.sin(angle_phi), distance * np.sin(angle_theta)), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, 1.0))
    model = glm.mat4(1.0)

    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    render_particles(particle_vertices, particle_shader_program, projection, view, model)

    # Swap buffers and poll events
    glfw.swap_buffers(window)
    glfw.poll_events()

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
ui = tk.Tk()
fov_slider = tk.Scale(ui, from_=10, to=120, orient='horizontal')
fov_slider.set(45)  # Set default FOV to 45 degrees
fov_slider.pack(pady=10)
distance_slider = tk.Scale(ui, from_=1, to=100, orient='horizontal')
distance_slider.set(10)  # Set default FOV to 45 degrees
distance_slider.pack(pady=10)
angle_slider = tk.Scale(ui, from_=-3, to=3, resolution=0.1, orient='horizontal')
angle_slider.set(0)  # Set default FOV to 45 degrees
angle_slider.pack(pady=10)
angle2_slider = tk.Scale(ui, from_=-3, to=3, resolution=0.1, orient='horizontal')
angle2_slider.set(0)  # Set default FOV to 45 degrees
angle2_slider.pack(pady=10)

shader_program_particle = load_shader()

# Enter the rendering loop, passing z_min and z_max

# Example particle data: Random particles in 3D space
particle_data = np.random.rand(3, 1000) * 10 - 5  # 100 random particles in the range [-5, 5]
particle_vertices = generate_particle_data(particle_data)
particle_VBO = setup_opengl_particles(particle_vertices)  # Particles

# Enter the rendering loop with particles
while not glfw.window_should_close(window):


    ui.update()
    render(window, particle_vertices, shader_program_particle, fov_slider.get(), angle_slider.get(), angle2_slider.get(), distance_slider.get())

    glfw.poll_events()


glfw.terminate()
