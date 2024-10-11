# This sample should be run after make libHEAAN.a file in lib folder

# All Target
all: clean TestImage TestNumber TestImageMultibatches conv conv_layer test_new new

# Tool invocations
TestImage:	
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
# g++ test.cpp -o TestHEAAN -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
# g++ encode.cpp -o Test -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
# g++ ntt_n.cpp -o Test -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	g++ test_image.cpp -o Test -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	@echo 'Finished building target: $@'
	@echo ' '

TestNumber:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ test_number.cpp -o Testt -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	@echo 'Finished building target: $@'
	@echo ' '

TestImageMultibatches:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ test_image_multibatches.cpp -o Testb -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	@echo 'Finished building target: $@'
	@echo ' '

conv:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ conv.cpp -o conv -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	@echo 'Finished building target: $@'
	@echo ' '

conv_layer:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ conv_layer.cpp -o conv_layer -L/usr/local/lib ../lib/libHEAAN.a -march=native -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	@echo 'Finished building target: $@'
	@echo ' '

test_new:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ -fsanitize=address test_new.cpp -o test_new -L/usr/local/lib ../lib/libHEAAN.a -march=native -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	@echo 'Finished building target: $@'
	@echo ' '

new:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ -fsanitize=address new.cpp -o new -L/usr/local/lib ../lib/libHEAAN.a -march=native -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	rm -rf Test* conv conv_layer test_new new

repeat:
	@for i in $$(seq 0 99); do \
		./Test ../../../FL/data/cifar10_images_batch_$$i.bin 300; \
	done