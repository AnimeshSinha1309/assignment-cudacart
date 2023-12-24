CXX_COMPILER=nvcc
SOURCE_FILES=cartpole/cartpole.cu cartpole/gpu.cu cartpole/image.cu cartpole/generate.cu
INCLUDE_PATHS=-I/usr/include/opencv4/
LINKED_LIBRARIES=-lcudart -lnppc -lnppc_static -lnppisu_static -lnppif_static -lfreeimageplus -lopencv_core -lopencv_imgcodecs -lopencv_highgui
OUTPUT_EXECUTABLE=cartpole.out

build:
	$(CXX_COMPILER) -g -G $(SOURCE_FILES) $(LINKED_LIBRARIES) $(INCLUDE_PATHS) -o $(OUTPUT_EXECUTABLE)

run:
	./$(OUTPUT_EXECUTABLE)

all: build run

debug:
	set cuda software_preemption on
	export CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1
	cuda-gdb ./$(OUTPUT_EXECUTABLE)
