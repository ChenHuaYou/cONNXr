CFLAGS=-fsanitize=undefined,address,leak
INCLUDE=-I../../include/ -I../../protobuf/ -I./include/
LIBS=-lconnxr -lm
build:
	gcc $(CFLAGS) $(INCLUDE) -g -o3 -std=c99 -o example example.c -L. $(LIBS)
clean:
	rm -f *.o *.a *.gch
