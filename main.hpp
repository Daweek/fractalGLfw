// General Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

// OpenGL Graphics related
#include <GL/glew.h>
#include <GLFW/glfw3.h>
// For OpenGL and Shaders utilities
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "shader.hpp"
#include "text2D.hpp"
#include "texture.hpp"

// Includes for CUDA and InteropGL
#include <cuda.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>

// QT temporal
#include <QtCore/QFile>
#include <QtGui/QMatrix2x2>
#include <QtGui/QVector2D>

// Defines for the Fractal
#define MIN_THREADS 1
#define DEFAULT_THREADS 512
#define MAX_THREADS 1024

#define MIN_POINTS 1
#define DEFAULT_POINTS 2500000
#define MAX_POINTS 5000000

//  Globals 
int 	 rendermode;
struct timeval time_v;
float	 g_cudakernltimer = 0;
double md_time,md_time0;
double disp_time,disp_time0;
double timeb,time0b;
double sock_time,sock_time0;


GLFWwindow* WindowID;
bool	g_renderLoopContinue = true;
static unsigned int winW = 1024, winH = 1024;

CUdevice g_devgpu;
cudaDeviceProp g_devprop;

GLuint 	g_mapVBO;
GLuint	m_uiFboTexture;
GLuint	m_uiFboDepth;
GLuint	m_uiFboFramBuff;

GLenum	m_eDrawBuffers[1];

GLuint VertexArrayID;

// For FRACTALS related
GLuint fracShaderID;
GLuint uScaling,uTranslation,uNumMappings;

QMatrix2x2 scalingMatrix;
QVector2D  translationVector;

typedef struct mp
{
	float x, y; // translation vertex
	float a, b, c, d; // scaling/rotation matrix
	float p; // mapping probability
} mapping;

mapping *m_mappings;

extern "C" void malloCUDA(mapping *);
extern "C" void generateFractal();
extern "C" void freeCUDA();

 // Reserve some memory for mappings
int numBlocks = 1;
int blockSize = 1024;
int numMappings = 0;
int numPoints = 1000000;
float kernel_mili = 0.0f;


void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  
 if( action == GLFW_PRESS ){

	 if(key == '?'){
	     printf("!   : initilize\n");
	     printf("q,Q : quit program\n");
	     printf("i,I : (print information of postion and angle)\n");
	     printf("r   : make radius small\n");
	     printf("R   : make radius large\n");
	     printf("d   : make ditail donw\n");
	     printf("D   : make ditail up\n");
	     printf("s   : md_step--\n");
	     printf("S   : md_step++\n");
	     printf("t,T : temp += 100\n");
	     printf("g,G : temp -= 100\n");
	     printf("y,Y : temp += 10\n");
	     printf("h,H : temp -= 10\n");
	     printf("z,Z : stop/restart\n");
	     printf("0-9 : chage particle number\n");
	   }

	
	 if(key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
		 g_renderLoopContinue = false;
   
 }
}

void DrawText(){

	char str_buf[256];
	int texX,texY,texSize;

	texSize	= 10*3;
	texX 		= 10*2;
	texY		= 550;

	sprintf(str_buf,"Fractals Rendering GPU");
	printText2D(str_buf, texX, texY, texSize);

  texY		-= texSize*16;
	sprintf(str_buf,"%.8f s",(disp_time-disp_time0));
	printText2D(str_buf, texX, texY, texSize);
	
	texY		-= texSize*1;
	sprintf(str_buf,"%.3f frm/s",1./(disp_time-disp_time0));
	printText2D(str_buf, texX, texY, texSize);
	
	texY		-= texSize*1;
	sprintf(str_buf,"%.3f ms -> Kernel<< >>",kernel_mili);
	printText2D(str_buf, texX, texY, texSize);

}


void display()
{

  // Prepare first window clearing color and enabling OpenGL states/behaviour
	glClearColor(0.0, 0.0,0.0,0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(fracShaderID);
	glUniform1i(uNumMappings,numMappings);
	glUniformMatrix2fv(uScaling,1,GL_TRUE,(float*)&scalingMatrix);
	glUniform2fv(uTranslation,1,(float*)&translationVector);

	// Draw Fractals
	{
		glBindBuffer(GL_ARRAY_BUFFER, g_mapVBO);
		glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE, 4 * sizeof(float),(void*)0);
		glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE, 4 * sizeof(float),(void*)(2*sizeof(float)));
		
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glPointSize(1.0);
		glDrawArrays(GL_POINTS,0,numPoints);
		
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		
	}

	// Take the time between each frame
	gettimeofday(&time_v,NULL);
	disp_time0 = disp_time;
	disp_time = (time_v.tv_sec + time_v.tv_usec / 1000000.0);
	// Draw Text
	DrawText();

	glfwSwapBuffers(WindowID);
	glfwPollEvents();

}

