# This sample should be run after make libHEAAN.a file in lib folder

# All Target
all: clean test test_new test_layer test_scheme new
# Tool invocations
test:	
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ test.cpp -o TestHEAAN -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	@echo 'Finished building target: $@'
	@echo ' '

test_new:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ -fsanitize=address test_new.cpp -o test_new -L/usr/local/lib ../lib/libHEAAN.a -march=native -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	@echo 'Finished building target: $@'
	@echo ' '

test_layer:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ -fsanitize=address test_layer.cpp -o test_layer -L/usr/local/lib ../lib/libHEAAN.a -march=native -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	@echo 'Finished building target: $@'
	@echo ' '

test_scheme:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ -fsanitize=address test_scheme.cpp -o test_scheme -L/usr/local/lib ../lib/libHEAAN.a -march=native -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	@echo 'Finished building target: $@'
	@echo ' '

new:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ new.cpp -o new -L/usr/local/lib ../lib/libHEAAN.a -march=native -lntl -lgmp -lm -std=c++11 -O3 -I/../src -pthread
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	rm -rf test Test* test_new test_layer test_scheme new

cleanall: clean
	@read -p "This will clear Public Key at ./serkey, Ciphertext checkpoints at ./cipher and ./temp. Proceed? (Y/n)" mainmenuinput; \
		if [ "$$mainmenuinput" = "Y" ] || [ "$$mainmenuinput" = "y" ] || [ -z "$$mainmenuinput" ]; then \
			rm -rf ./serkey ./cipher ./temp; \
			echo "Done"; \
		fi