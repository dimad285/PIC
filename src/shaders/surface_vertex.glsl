#version 330 core
layout (location = 0) in vec3 position;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform float zMin;
uniform float zMax;

out vec3 FragPos;
out float Height;

void main()
{
    FragPos = vec3(model * vec4(position, 1.0));
    Height = (position.z + 1.0) * 0.5;  // Convert back from [-1,1] to [0,1] for coloring
    gl_Position = projection * view * model * vec4(position, 1.0);
}