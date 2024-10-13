import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np

vertex_shader = """
#version 330
in vec3 position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 fragColor;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    fragColor = vec3(0.5 + position.z * 0.5, 0.5 - position.z * 0.5, 0.5);
    gl_PointSize = 2.0;
}
"""

fragment_shader = """
#version 330
in vec3 fragColor;
out vec4 outColor;

void main()
{
    outColor = vec4(fragColor, 1.0);
}
"""

def create_shader_program(vertex_shader, fragment_shader):
    shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )
    return shader

def create_surface(nx, ny):
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xv, yv = np.meshgrid(x, y)
    zv = np.zeros((ny, nx))
    vertices = np.dstack((xv, yv, zv)).reshape(-1, 3).astype(np.float32)
    return vertices

def create_mvp_matrices():
    model = np.identity(4, dtype=np.float32)
    view = np.identity(4, dtype=np.float32)
    view[3, 2] = -3.0
    projection = np.identity(4, dtype=np.float32)
    fov = 45.0
    aspect = 800.0 / 600.0
    near = 0.1
    far = 100.0
    projection[1, 1] = 1 / np.tan(np.radians(fov) / 2)
    projection[0, 0] = projection[1, 1] / aspect
    projection[2, 2] = -(far + near) / (far - near)
    projection[2, 3] = -2 * far * near / (far - near)
    projection[3, 2] = -1
    return model, view, projection

def main():
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "PIC 3D Surface Plot", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    shader = create_shader_program(vertex_shader, fragment_shader)
    glUseProgram(shader)

    # Create surface
    vertices = create_surface(100, 100)

    # Create and bind a Vertex Array Object
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # Create and bind a Vertex Buffer Object
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)

    # Set the vertex attribute pointers
    position_loc = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position_loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position_loc)

    # Create MVP matrices
    model, view, projection = create_mvp_matrices()

    # Get uniform locations
    model_loc = glGetUniformLocation(shader, "model")
    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")

    # Set uniform variables
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    glEnable(GL_DEPTH_TEST)
    glPointSize(2.0)

    t = 0
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Update z values to create animation
        vertices[:, 2] = 0.3 * np.sin(2 * np.pi * (vertices[:, 0] + vertices[:, 1] + t))
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

        glDrawArrays(GL_POINTS, 0, len(vertices))

        glfw.swap_buffers(window)
        glfw.poll_events()

        t += 0.02

    glfw.terminate()

if __name__ == "__main__":
    main()
    print("Program completed.")