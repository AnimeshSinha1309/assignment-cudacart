CXX_COMPILER=nvcc
SOURCE_FILES=cartpole/cartpole.cu cartpole/gpu.cu cartpole/image.cu
LINKED_LIBRARIES=-lcudart -lnppc -lnppc_static -lnppisu_static -lnppif_static
OUTPUT_EXECUTABLE=cartpole.out

build:
	$(CXX_COMPILER) $(SOURCE_FILES) $(LINKED_LIBRARIES) -o $(OUTPUT_EXECUTABLE)

run:
	./$(OUTPUT_EXECUTABLE)

all: build run
