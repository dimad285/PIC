from OpenGL.GL.shaders import compileProgram, compileShader
import glfw
from OpenGL.GL import *
import numpy as np
import glm

def initialize_window():
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

def create_shader_program():
    vertex_shader = """
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
    
    fragment_shader = """
    #version 330 core

    in float vertexZ;  // Receive the Z-coordinate from the vertex shader
    out vec4 fragColor;

    uniform float zMin;  // Minimum Z value (to normalize)
    uniform float zMax;  // Maximum Z value (to normalize)

    void main() {
        // Normalize the Z-coordinate to the range [0, 1]
        float normalizedZ = (vertexZ - zMin) / (zMax - zMin);

        // Interpolate between blue (low Z) and red (high Z)
        vec3 lowColor = vec3(0.0, 0.0, 1.0);  // Blue
        vec3 highColor = vec3(1.0, 0.0, 0.0);  // Red

        vec3 finalColor = mix(lowColor, highColor, normalizedZ);
        
        fragColor = vec4(finalColor, 1.0);  // Final color with full opacity
    }
    """
    
    # Compile shaders
    shader_program = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )
    
    return shader_program

def setup_opengl(vertices):
    # Create a Vertex Buffer Object (VBO) to store the vertex data on the GPU
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Define the structure of the vertex data
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)


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


def generate_surface_data():
    # Generate a grid of points (X, Y) and corresponding Z values for the surface
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))  # Example surface (circular wave)
    return X, Y, Z



def render_loop(window, vertices, shader_program, z_min, z_max):
    # Enable depth testing for 3D rendering
    glEnable(GL_DEPTH_TEST)

    # Use the shader program
    glUseProgram(shader_program)

    # Set up projection and view matrices
    projection = glm.perspective(glm.radians(45.0), 800/600, 0.1, 100.0)
    view = glm.lookAt(glm.vec3(10.0, 10.0, 10.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, 10.0))
    model = glm.mat4(1.0)

    # Get uniform locations
    proj_loc = glGetUniformLocation(shader_program, "projection")
    view_loc = glGetUniformLocation(shader_program, "view")
    model_loc = glGetUniformLocation(shader_program, "model")
    z_min_loc = glGetUniformLocation(shader_program, "zMin")
    z_max_loc = glGetUniformLocation(shader_program, "zMax")

    while not glfw.window_should_close(window):
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Update the view and projection matrices
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))

        # Set the Z-min and Z-max values
        glUniform1f(z_min_loc, z_min)
        glUniform1f(z_max_loc, z_max)

        # Draw the surface (as triangles)
        glDrawArrays(GL_TRIANGLES, 0, len(vertices) // 3)

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()



window = initialize_window()

# Generate surface data and vertices
X, Y, Z = generate_surface_data()

z_min = np.min(Z)
z_max = np.max(Z)

vertices = generate_vertices(X, Y, Z)

# Set up OpenGL buffers and shaders
setup_opengl(vertices)
shader_program = create_shader_program()

# Enter the rendering loop, passing z_min and z_max
render_loop(window, vertices, shader_program, z_min, z_max)