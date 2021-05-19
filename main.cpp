#include "main.hpp"


void initFractal(){
	// Init Fracta from File
	
	//QString filename = "./fractals/barnsleyfern.frac";
	QString filename = "./fractals/durerpentagon.frac";

	QFile file(filename);
  if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
    qFatal("ERROR: file %s not found - exiting.", qPrintable(filename));
	
	QTextStream in(&file);

	// Read in name of fractal
	QString name = in.readLine();
	std::cout<<name.toUtf8().constData()<<"\n";
	

	// Read in number of mappings
	in >> numMappings;

	// Allocate mem for mappings
	m_mappings = (mapping*) malloc(numMappings*sizeof(mapping));	
	// Read in all given mappings
	for(int i = 0; i < numMappings; i++)
	{
			float a, b, c, d, x, y, p;
			in >> a >> b >> c >> d >> x >> y >> p;
			m_mappings[i] = {x, y, a, b, c, d, p};
		
	}

	// Read in scaling and translation factors for rendering
	float scaleX, scaleY, translationX, translationY;
	in >> scaleX >> scaleY >> translationX >> translationY;

	float scalingValues[] = {scaleX, 0.0f, 0.0f, scaleY};
	scalingMatrix = QMatrix2x2(scalingValues);
	translationVector = QVector2D(translationX, translationY);

}

void initGL(int argc, char **argv)
{
	glfwInit();
	glfwWindowHint(GLFW_SAMPLES, 4);
	
	WindowID = glfwCreateWindow( winW, winH, "FractalCreator...", NULL, NULL);
	glfwMakeContextCurrent(WindowID);

	//	Glew initialization to have access to the extensions
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK)
		assert(!"Failed to initialize GLEW.\n");
	if (!glewIsSupported("GL_EXT_framebuffer_object"))
		assert(!"The GL_EXT_framebuffer_object extension is required.\n");

	glfwSetWindowPos(WindowID, 5400, 2100);
	glfwSetKeyCallback(WindowID, keyboard);
	  
	glfwSetInputMode(WindowID,GLFW_STICKY_KEYS,GLFW_STICKY_KEYS);
	glfwSetInputMode(WindowID, GLFW_CURSOR, GLFW_CURSOR_NORMAL);


	// Create Color Attachment for frame buffer
	glGenTextures(1, &m_uiFboTexture);
	glBindTexture(GL_TEXTURE_RECTANGLE, m_uiFboTexture);
	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, winW, winH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// Create Depth Attachment for frame buffer
	glGenRenderbuffers( 1, &m_uiFboDepth );
	glBindRenderbuffer( GL_RENDERBUFFER, m_uiFboDepth );
	glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT, winW, winH);
	glBindRenderbuffer( GL_RENDERBUFFER, 0 );

	// Create Frame buffer
	glGenFramebuffers(1, &m_uiFboFramBuff);
	glBindFramebuffer(GL_FRAMEBUFFER, m_uiFboFramBuff);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_uiFboTexture, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_uiFboDepth);

	// Check Frame buffer status
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		assert(!"Framebuffer is incomplete.\n");
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Load Shader for Text
	initText2D("Holstein.DDS");
	
	// Load Fractal shaders
	fracShaderID			= LoadShaders("shaders/shader.vert", "shaders/shader.frag");
	uScaling					= glGetUniformLocation(fracShaderID, "scaling");
	uTranslation			= glGetUniformLocation(fracShaderID, "translation");
	uNumMappings			= glGetUniformLocation(fracShaderID, "numMappings");
}

void initCUDA(){

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGetDevice(&g_devgpu));
	
	printf("Device %d: %s is used!\n", g_devgpu, g_devprop.name);
	
	// Query device properties
	cudaDeviceProp prop;
	int driverVersion, runtimeVersion;
	cudaGetDeviceProperties(&prop, g_devgpu);
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);

	// Print device properties
	printf("\tDevice Name: %s\n", prop.name);
	printf("\tCUDA Driver Version / Runtime Version: %d.%d / %d.%d\n",
					driverVersion / 1000, (driverVersion % 100) / 10,
					runtimeVersion / 1000, (runtimeVersion % 100) / 10);
	printf("\tCompute Capability: %d.%d\n", prop.major, prop.minor);
	printf("\tTotal Global Memory: %ld bytes\n", prop.totalGlobalMem);
	printf("\tNumber of Multiprocessors: %d\n", prop.multiProcessorCount);
	printf("\tMaximum Threads per Multiprocessor: %d\n",
					prop.maxThreadsPerMultiProcessor);
	printf("\tTotal Number of Threads: %d\n", prop.multiProcessorCount *
					prop.maxThreadsPerMultiProcessor);
	printf("\tMaximum Threads per Block: %d\n", prop.maxThreadsPerBlock);

}

int main(int argc, char **argv)
{
  // TODO catch main arguments
	std::cout<<"Init FractalCreator...\n";
  
	initCUDA();

	initFractal();
	
	initGL(argc, argv);
	
 	malloCUDA(m_mappings);

	// Options for OpenGL rendering
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

	glGenVertexArrays(1,&VertexArrayID);
	glBindVertexArray(VertexArrayID);

	while(g_renderLoopContinue && !glfwWindowShouldClose(WindowID)){

		generateFractal();

		display();

  }

	glfwDestroyWindow(WindowID);
  glDeleteVertexArrays(0,&VertexArrayID);

	freeCUDA();

	glfwTerminate();

  printf("Exit-->0\n");
  return 0;
}


