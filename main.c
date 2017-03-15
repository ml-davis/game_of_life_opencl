#include "ErrorMessages.c"

#include <CL/cl.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 1024
#define HEIGHT 768
#define NUM_SPECIES 10

int *grid = NULL;
cl_device_id *devices;
cl_command_queue cmdQueue;
cl_context context;
cl_uint numDevices;
cl_platform_id *platforms;
cl_program program;
size_t globalWorkSize[1];
cl_kernel kernel;
cl_mem temp_buffer;
size_t datasize = sizeof(int) * WIDTH * HEIGHT;
cl_mem buffer;

int loop_counter = 0;

// OpenCL kernel to perform an element-wise add of 2 arrays
const char *programSource =
"__kernel																							\n"
"void update_grid(__global int *buffer, __global int *temp_buffer) {								\n"
"	int idx = get_global_id(0);																		\n"
"	int species = buffer[idx];																		\n"
"																									\n"
"	int x = idx % 1024;																				\n"
"	int y = idx / 1024;																				\n"
"																									\n"
"	int count;																						\n"
"	if (species != 0) {	// if cell isn't dead														\n"
"		count = 0;																					\n"
"																									\n"
"		// iterate over 3x3 grid centered on (x,y)													\n"
"		for (int i = x - 1; i <= x + 1; i++) {														\n"
"			for (int j = y - 1; j <= y + 1; j++) {													\n"
"				// check only if cell isn't current cell (x,y) AND cell is not out of bounds		\n"
"				if ((i != x || j != y) && (i >= 0 && j >= 0 && i < 1024 && j < 768)) {				\n"
"					int cell = j * 1024 + i;														\n"
"					if (buffer[cell] == species) {													\n"
"						count++;																	\n"
"					}																				\n"
"				}																					\n"
"			}																						\n"
"		}																							\n"
"		if (count < 2 || count > 3) {																\n"
"			temp_buffer[idx] = 0; // kill cell														\n"
"		}																							\n"
"	} else { // if cell is dead																		\n"
"		for (int s = 1; s <= 10; s++) { // for each species											\n"
"			count = 0;																				\n"
"			for (int i = x - 1; i <= x + 1; i++) {													\n"
"				for (int j = y - 1; j <= y + 1; j++) {												\n"
"					// check only if cell isn't current cell (x,y) AND cell is not out of bounds	\n"
"					if ((i != x || j != y) && (i >= 0 && j >= 0 && i < 1024 && j < 768)) {			\n"
"						int cell = j * 1024 + i;													\n"
"						if (buffer[cell] == s) {													\n"
"							count++;																\n"
"						}																			\n"
"					}																				\n"
"				}																					\n"
"			}																						\n"
"			if (count == 3) {																		\n"
"				temp_buffer[idx] = s; // spawn cell													\n"
"			}																						\n"
"		}																							\n"
"	}																								\n"
"}																									\n";

void set_cell(int x, int y, int s) {
    int index = WIDTH * y + x;
    grid[index] = s;
}

int get_cell(int x, int y) {
    int index = WIDTH * y + x;
    return grid[index];
}

// randomly spawns creatures in shotgun patterns on board
void initialize_grid() {
    srand(time(NULL));
    grid = (int *) malloc(WIDTH * HEIGHT * sizeof(int));

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            set_cell(j, i, 0);
        }
    }

    for (int i = 1; i <= NUM_SPECIES; i++) {

        int square_size = WIDTH * .20;

        // fill ~20% of square
        int number_of_squares = (int) floor((square_size * square_size) * 0.18);
        int distance_from_edge = square_size + 2;

        // choose random target on board, at least specified distance from edges
        int x_target = (rand() % (WIDTH - (distance_from_edge * 2 - 1))) + distance_from_edge;
        int y_target = (rand() % (HEIGHT - (distance_from_edge * 2 - 1))) + distance_from_edge;

        set_cell(x_target, y_target, i);

        // pick number_of_squares within (square_size x square_size) square centered on target
        int rand_x;
        int rand_y;
        for (int j = 0; j < number_of_squares; j++) {
            rand_x = x_target + ((rand() % (square_size + 1)) - (square_size/2));
            rand_y = y_target + ((rand() % (square_size + 1)) - (square_size/2));
            set_cell(rand_x, rand_y, i);
        }
    }
}

// sets the color that OpenGL will draw with
void set_color(int species) {
    switch(species) {
        case 1:	glColor3f(1.0f, 0.0f, 0.0f); break; // RED
        case 2:	glColor3f(0.0f, 1.0f, 0.0f); break; // GREEN
        case 3:	glColor3f(0.1f, 0.2f, 1.0f); break; // BLUE
        case 4:	glColor3f(1.0f, 1.0f, 0.0f); break; // YELLOW
        case 5:	glColor3f(1.0f, 0.0f, 1.0f); break; // MAGENTA
        case 6:	glColor3f(0.0f, 1.0f, 1.0f); break; // CYAN
        case 7:	glColor3f(1.0f, 1.0f, 1.0f); break; // WHITE
        case 8:	glColor3f(1.0f, 0.5f, 0.0f); break; // ORANGE
        case 9:	glColor3f(0.5f, 0.5f, 0.5f); break; // GREY
        case 10:glColor3f(0.4f, 0.0f, 1.0f); break; // VIOLET
        default:glColor3f(0.0f, 0.0f, 0.0f);		// BLACK
    }
}

// places a square at (x, y). Must be nested in glBegin() <-> glEnd() tags
void draw_square(int x, int y) {
    if (x < 0 || y < 0 || x >= WIDTH || y >= HEIGHT) {
        printf("Invalid range in draw_square. (%d, %d) out of range", x, y);
        exit(1);
    }

    glVertex2f(x, y);
    glVertex2f(x + 1, y);
    glVertex2f(x + 1, y + 1);
    glVertex2f(x, y + 1);
}

int k = 0;

void initialize_opencl() {

	/******************************************************
	 * Discover and initialize the platforms
	 *****************************************************/
	cl_uint numPlatforms = 0;

	// Use clGetPlatformIDs() to retrieve the number of platforms
	cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS) {
		printf("Error while initializing platforms\n");
		printf("%s\n", getErrorString(status));
	}

	// Allocate enough space for each platform
	platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));

	// Fill in platforms with clGetPlatformIDs()
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS) {
		printf("Error while initializing platforms\n");
		printf("%s\n", getErrorString(status));
	}

	/******************************************************
	 * Discover and initialize the devices
	 *****************************************************/

	// use clGetDeviceIDs() to retrieve # of devices
	status = clGetDeviceIDs(
		platforms[0],
		CL_DEVICE_TYPE_ALL,
		0,
		NULL,
		&numDevices
	);
	if (status != CL_SUCCESS) {
		printf("Error discovering and initializing the devices\n");
		printf("%s\n", getErrorString(status));
	}

	// Allocate enough space for each device
	devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));

	// Fill in devices with clGetDeviceIDs()
	status = clGetDeviceIDs(
		platforms[0],
		CL_DEVICE_TYPE_ALL,
		numDevices,
		devices,
		NULL
	);
	if (status != CL_SUCCESS) {
		printf("Error discovering and initializing the devices\n");
		printf("%s\n", getErrorString(status));
	}

	/******************************************************
	 * Create a context
	 ******************************************************/
	// Create a context and associate it with the devices
	context = clCreateContext(
		NULL,
		numDevices,
		devices,
		NULL,
		NULL,
		&status
	);
	if (status != CL_SUCCESS) {
		printf("Error while creating context\n");
		printf("%s\n", getErrorString(status));
	}

	/******************************************************
	 * Create a command queue
	 *****************************************************/
	// Create a command queue and associate it with the device you
	// want to execute on
	cmdQueue = clCreateCommandQueueWithProperties(
		context,
		devices[0],
		0,
		&status
	);
	if (status != CL_SUCCESS) {
		printf("Error while creating command queue\n");
		printf("%s\n", getErrorString(status));
	}

	/******************************************************
	 * Create device buffers
	 *****************************************************/
	// create buffer object for device that will hold the host grid data
	buffer = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		datasize,
		NULL,
		&status
	);
	if (status != CL_SUCCESS) {
		printf("Error while creating device buffers\n");
		printf("%s\n", getErrorString(status));
	}

	temp_buffer = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		datasize,
		NULL,
		&status
	);
	if (status != CL_SUCCESS) {
		printf("Error while creating device buffers\n");
		printf("%s\n", getErrorString(status));
	}

	/******************************************************
	 * Create and compile the program
	 *****************************************************/
	program = clCreateProgramWithSource(
		context,
		1,
		(const char**)&programSource,
		NULL,
		&status
	);
	if (status != CL_SUCCESS) {
		printf("Error while compiling kernel program\n");
		printf("%s\n", getErrorString(status));
	}

	// compile the program for the devices
	status = clBuildProgram(
		program,
		numDevices,
		devices,
		NULL,
		NULL,
		NULL
	);
	if (status != CL_SUCCESS) {
		printf("Error while compiling kernel program\n");
		printf("%s\n", getErrorString(status));
	}

	/******************************************************
	 * Create the kernel
	 *****************************************************/
	// create a kernel from the update_grid function
	kernel = clCreateKernel(program, "update_grid", &status);
	if (status != CL_SUCCESS) {
		printf("Error while creating kernel\n");
		printf("%s\n", getErrorString(status));
	}

	/******************************************************
	 * Set the kernel arguments
	 *****************************************************/
	status |= clSetKernelArg(
		kernel,
		0,
		sizeof(cl_mem),
		&buffer
	);
	if (status != CL_SUCCESS) {
		printf("Error while setting kernel arguments\n");
		printf("%s\n", getErrorString(status));
	}

	status |= clSetKernelArg(
		kernel,
		1,
		sizeof(cl_mem),
		&temp_buffer
	);
	if (status != CL_SUCCESS) {
		printf("Error while setting kernel arguments\n");
		printf("%s\n", getErrorString(status));
	}

	/******************************************************
	 * Configure the work-item structure
	 *****************************************************/
	// define an index space (global work size) of work items for execution and workgroup size
	globalWorkSize[0] = WIDTH*HEIGHT;
}

void update() {
	printf("loop %d\n", loop_counter++);

	// Use this to check the output of each API call
	cl_int status;

	/******************************************************
	 * Write host data to device buffers
	 *****************************************************/
	printf("Loop %d\n", loop_counter++);

	// write host data to device buffers
	status = clEnqueueWriteBuffer(
		cmdQueue,
		buffer,
		CL_FALSE,	// blocking_write
		0,			// offset
		datasize,
		grid,
		0,			// num events in wait list
		NULL,		// wait list
		NULL		// event
	);
	if (status != CL_SUCCESS) {
		printf("Error while enqueuing buffer\n");
		printf("%s\n", getErrorString(status));
	}

	status = clEnqueueWriteBuffer(
		cmdQueue,
		temp_buffer,
		CL_FALSE,	// blocking_write
		0,			// offset
		datasize,
		grid,
		0,			// num events in wait list
		NULL,		// wait list
		NULL		// event
	);
	if (status != CL_SUCCESS) {
		printf("Error while enqueuing buffer\n");
		printf("%s\n", getErrorString(status));
	}

	/******************************************************
	 * Enqueue the kernel for execution
	 *****************************************************/
	status = clEnqueueNDRangeKernel(
		cmdQueue,
		kernel,
		1, 					// num dimensions
		NULL,
		globalWorkSize,
		NULL,
		0,
		NULL,
		NULL
	);
	if (status != CL_SUCCESS) {
		printf("Error while enqueuing kernel for execution\n");
		printf("%s\n", getErrorString(status));
	}

	/******************************************************
	 * Read the output buffer back to the host
	 *****************************************************/
	clEnqueueReadBuffer(
		cmdQueue,
		temp_buffer,
		CL_TRUE,
		0,
		WIDTH*HEIGHT*sizeof(int),
		grid,
		0,
		NULL,
		NULL
	);
}

// Infinite loop, repeatedly draws board and updates
void display() {
	for (;;) {
		glBegin(GL_QUADS);
		for (int i = 0; i < HEIGHT; i++) {
			for (int j = 0; j < WIDTH; j++) {
				int s = get_cell(j, i);
				set_color(s);
				draw_square(j, i);
			}
		}
		glEnd();
		glFlush();

		update();
	}
}

int main(int argc, char *argv[]) {

	// Initialize host grid and opencl

	printf("Initializing grid\n");
    initialize_grid();
	printf("Finished initializing grid\n");

	printf("Initializing opencl\n");
	initialize_opencl();
	printf("Finshed initializing opencl\n");

	// Initialize opengl
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Game of Life - OpenCL");
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0f, WIDTH, HEIGHT, 0.0f, -1.0f, 1.0f);
	glutDisplayFunc(display); // enter display function (loops here)
	glutMainLoop();

	// Release resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseContext(context);
	free(platforms);
	free(devices);
	free(grid);
}
