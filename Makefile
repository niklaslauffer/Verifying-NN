CC = g++
CFLAGS = -std=c++11 -no-pie
INCLUDE = -I/usr/local/include 
LFLAGS = -L/usr/local/lib
SRCS = DAG.cpp build_network.cpp
LIBS = -lyaml-cpp
MAIN = mountain_car.cpp
NAME = mountain_car

main :
	$(CC) $(CFLAGS) -o $(NAME) $(MAIN) $(SRCS) $(INCLUDE) $(LFLAGS) $(LIBS)

clean :
	rm -f core $(NAME)
