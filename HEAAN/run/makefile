# This sample should be run after make libHEAAN.a file in lib folder

# All Target
all: clean TestHEAAN

# Tool invocations
TestHEAAN:
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
# g++ test.cpp -o TestHEAAN -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
# g++ encode.cpp -o Test -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	g++ ntt_n.cpp -o Test -L/usr/local/lib ../lib/libHEAAN.a -lntl -lgmp -lm -std=c++11 -O2 -I/../src -pthread
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	rm -rf TestHEAAN

repeat:
	@echo 'Running ./Test 10 times...'
	@for i in $$(seq 1 10); do \
		echo 'Running iteration' $$i; \
		./Test; \
		echo 'Iteration $$i finished'; \
	done