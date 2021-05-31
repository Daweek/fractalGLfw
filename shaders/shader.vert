#version 330 core

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 color;

out vec4 vColor;

uniform int numMappings;
//uniform mat2 scaling;
//uniform vec2 translation;

/*
 * Converts HSV value to RGB
 * Code from http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
 * Running in vertex shader over fragment shader for performance
 */
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main()
{
    vec2 translation = vec2(0.0f,0.0f);
    mat2 scaling = mat2(0.20f,0.0f,0.0f,0.20f);


    vec2 transformedPosition = (scaling * position + translation)*vec2(1.0,-1.0);
    //vec2 transformedPosition = position;
    gl_Position = vec4(transformedPosition, 0.0, 1.0);
    //gl_Position = vec4(0.0,0.0,0.0, 1.0);
    //gl_PointSize = 50.0;

    //vec3 hsvColor = vec3(color.y/numMappings, 1.0, 1.0);
    //vec3 rgbColor = hsv2rgb(hsvColor);

    // Gray color
    vec3 rgbColor = vec3(0.5,0.5,0.5);
    
    vColor = vec4(rgbColor, 1.0);
}
