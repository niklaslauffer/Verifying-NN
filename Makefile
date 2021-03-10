CURRENT_DIR = $(shell pwd)
CC = g++
CFLAGS = -std=c++11 -no-pie -Wall -frounding-math
INCLUDE = -I/usr/local/include -I$(CURRENT_DIR)/aaflib-0.1 -I.
LFLAGS = -L/usr/local/lib -L$(CURRENT_DIR)/aaflib-0.1
SRCS = DAG.cpp build_network.h
LIBS = -lyaml-cpp -laaf -lprim -lgsl -llapack -lblas -lstdc++
# MAIN = wildlife.cpp
# NAME = wildlife
MAIN = wildlife_af.cpp
NAME = wildlife_af

main :
	$(CC) $(CFLAGS) -o $(NAME) $(MAIN) $(SRCS) $(INCLUDE) $(LFLAGS) $(LIBS)

clean :
	rm -f core $(NAME)
