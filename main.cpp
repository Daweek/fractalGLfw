#include "main.hpp"


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
	//uScaling					= glGetUniformLocation(fracShaderID, "scaling");
	//uTranslation			= glGetUniformLocation(fracShaderID, "translation");
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

void saveImage(){
	// Make the BYTE array, factor of 3 because it's RBG.
	BYTE* pixels = new BYTE[3 * winW * winH];

	glReadPixels(0, 0, winW, winH, GL_RGB, GL_UNSIGNED_BYTE, pixels);

	// Convert to FreeImage format & save to file
	FIBITMAP* image = FreeImage_ConvertFromRawBits
							(pixels, winW, winH, 3 * winW, 24, 0x0000FF, 0xFF0000, 0x00FF00, false);
	
	FreeImage_Save(FIF_BMP, image, "./data/test.bmp", 0);

	// Free resources
	FreeImage_Unload(image);
	delete [] pixels;
}

void savePNG(const char *filepath){

	//const char *filepath = "./data/map.png";

	int width, height;
	glfwGetFramebufferSize(WindowID, &width, &height);
	GLsizei nrChannels = 3;
	GLsizei stride = nrChannels * width;
	stride += (stride % 4) ? (4 - stride % 4) : 0;
	GLsizei bufferSize = stride * height;
	std::vector<char> buffer(bufferSize);
	glPixelStorei(GL_PACK_ALIGNMENT, 4);
	glReadBuffer(GL_FRONT);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
	stbi_flip_vertically_on_write(true);
	stbi_write_png(filepath, width, height, nrChannels, buffer.data(), stride);
}

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
	//std::cout<<name.toUtf8().constData()<<"\n";


	// Read in number of mappings
	in >> numMappings;

	// Allocate mem for mappings
	m_map = (mapping*) malloc(numMappings*sizeof(mapping));	
	// Read in all given mappings
	for(int i = 0; i < numMappings; i++)
	{
			float a, b, c, d, x, y, p;
			in >> a >> b >> c >> d >> x >> y >> p;
			m_map[i] = {x, y, a, b, c, d, p};
		
	}

	// Read in scaling and translation factors for rendering
	//float scaleX, scaleY, translationX, translationY;
	//in >> scaleX >> scaleY >> translationX >> translationY;

	//float scalingValues[] = {scaleX, 0.0f, 0.0f, scaleY};
	//scalingMatrix = QMatrix2x2(scalingValues);
	//translationVector = QVector2D(translationX, translationY);

}



void prepareInitValues(){

	float a,b,c,d,e,f,prob;
	int param_size;
	float sum_proba;
	
	a = b = c = d = e = f = prob = sum_proba = 0.0f;

	param_size = nc::random::randInt<int>(2,8);
	numMappings = param_size;
	//std::cout<<"Param_size:"<<param_size<<std::endl;

	m_map = (mapping*)malloc(param_size * sizeof(mapping));
	

	for (int i=0;i <param_size;i++){
		
		a = b = c = d = e = f = 0.0f;
		auto param_rand = nc::random::uniform<float>({1,6},-1.0f,1.0f);
		//param_rand.print();
		a = param_rand(0,0);
		b = param_rand(0,1);
		c = param_rand(0,2);
		d = param_rand(0,3);
		e = param_rand(0,4);
		f = param_rand(0,5);
		
		prob = abs(a*d - b*c);
		sum_proba += prob;

		m_map[i] = {e, f, a, b, c, d, prob};
		//cout<<m_map[i].x<<endl;
	}

	for (int i=0;i <param_size;i++){
		m_map[i].p /= sum_proba;
		//cout<<m_map[i].p<<endl;
	}

	//std::cout<<std::endl<<std::endl<<std::endl<<std::endl;
}



void write_csv(std::string filename, mapping *m, int numMaps){
    // Make a CSV file with one column of integer values
    // filename - the name of the file
    // colname - the name of the one and only column
    // vals - an integer vector of values
    
    // Create an output filestream object
		std::cout.setf(std::ios::scientific);
    std::ofstream myFile(filename);
    
		// Send the column name to the stream
    //myFile << colname << "\n";
    
    // Send data to the stream
    for(int i = 0; i < numMaps; ++i)
    {
        myFile<<scientific<< m[i].a <<","
													<< m[i].b <<","
													<< m[i].c <<","
													<< m[i].d <<","
													<< m[i].x <<","
													<< m[i].y <<","
													<< m[i].p <<"\n";
    }
    
		//cout<<"Writting csv..."<<endl;
    // Close the file
    myFile.close();
}

bool IsPathExist(const std::string &s)
{
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}

void initFromCSV(){

	stringstream ss;
	ss<<setw(5)<<setfill('0')<<to_string(file);
	string s = ss.str();

	string _path = "data/csv_rate0.2_category100/" + s + ".csv";
	cout<<_path<<endl;

	ifstream in(_path);
	vector<vector<double>> fields;

	if (in) {
			string line;

			while (getline(in, line)) {
					stringstream sep(line);
					string field;

					fields.push_back(vector<double>());

					while (getline(sep, field, ',')) {
							fields.back().push_back(stod(field));
					}
			}
	}

	cout.setf(ios::scientific);
	cout.precision(18);
	for (auto row : fields) {
			for (auto field : row) {

					cout << field << ' ';
			}

			cout << '\n';
	}

	// Allocate to 
	float a,b,c,d,e,f,prob;
	int param_size;
	float sum_proba;
	
	a = b = c = d = e = f = prob = sum_proba = 0.0f;

	param_size = fields.size();
	numMappings = param_size;
	std::cout<<"Param_size:"<<param_size<<std::endl;

	if(m_map != NULL)
		free(m_map);

	m_map = (mapping*)malloc(param_size * sizeof(mapping));
	

	for (int i=0;i <param_size;i++){
		
		a = b = c = d = e = f = 0.0f;
		
		//param_rand.print();
		a = fields[i][0];
		b = fields[i][1];
		c = fields[i][2];
		d = fields[i][3];
		e = fields[i][4];
		f = fields[i][5];
		prob = fields[i][6];
		
		//prob = abs(a*d - b*c);
		//sum_proba += prob;

		m_map[i] = {e, f, a, b, c, d, prob};
		//cout<<m_map[i].x<<endl;
	}
	
}

int main(int argc, char **argv)
{
  // TODO catch main arguments
	int num = 0;

	if(argc > 1){
		numOfClass = atoi(argv[1]);
	}

	std::cout<<"Init FractalCreator...\n";
  nc::random::seed(669);

	

	initCUDA();

	//initFractal();
	initFromCSV();
	//prepareInitValues();
	
	initGL(argc, argv);
	
	interopCUDA();
	
 	malloCUDA(m_map);

	// Options for OpenGL rendering
  
	glEnable(GL_POINTS);
	glDisable(GL_POINT_SMOOTH);
	glDisable(GL_MULTISAMPLE);
	
	glGenVertexArrays(1,&VertexArrayID);
	glBindVertexArray(VertexArrayID);


	ostringstream o;
	o.precision(2);
	o<<fixed<<dense;
	string outPathcsv = output + "csv_ratio"+o.str();
	string outPathimg = output + "rate"+o.str();

	filesystem::create_directory(outPathcsv);
	filesystem::create_directory(outPathimg);


#if 1

	while(g_renderLoopContinue && !glfwWindowShouldClose(WindowID)){

		if(g_generateFractal){
			//free(m_map);
			//prepareInitValues();
			initFromCSV();
			malloCUDA(m_map);
			generateFractal();
			numPixel();
			display();
		}
		else {
			generateFractal();
			display();
		}
  }
#else

	while(num < numOfClass){

		if(m_map != NULL)free(m_map);
		prepareInitValues();
		malloCUDA(m_map);
		generateFractal();
		display();

		if(numPixel() >= dense){

			stringstream ss;
			ss<<setw(5)<<setfill('0')<<to_string(num);
			string s = ss.str();
			//cout<<s<<endl;

			string filecsv = outPathcsv +"/" + s + ".csv";
			//cout<<filecsv<<endl;
			write_csv(filecsv,m_map,numMappings);
						
			string fileimg = outPathimg + "/" + s + ".png";
			//cout<<fileimg<<endl;
			savePNG(fileimg.c_str());

			cout<<"save: " << s << endl;
			num++;
		}
	}
#endif



	//string path = output + to_string(numOfClass);

	//if(IsPathExist("data/")){
		
		//write_csv("test.csv",m_map,numMappings);
	//}
	//saveImage();
	//savePNG();

	glfwDestroyWindow(WindowID);
  glDeleteVertexArrays(0,&VertexArrayID);

	freeCUDA();

	glfwTerminate();

  printf("Exit-->0\n");
  return 0;
}


