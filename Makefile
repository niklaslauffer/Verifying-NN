CC = g++
CFLAGS = -std=c++11 
INCLUDE = -I/usr/local/include 
LFLAGS = -L/usr/local/lib
SRCS = DAG.cpp build_network.cpp
LIBS = -lyaml-cpp
MAIN = main.cpp
NAME = test_network

main :
	$(CC) $(CFLAGS) -o $(NAME) $(MAIN) $(SRCS) $(INCLUDE) $(LFLAGS) $(LIBS)

clean :
	rm -f core $(NAME)
