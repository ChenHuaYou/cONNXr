CFLAGS=-fsanitize=undefined,address,leak
INCLUDE=-I../include/ -I../protobuf/ -I./include/
LIBDIR=-L../build/
LIBS=-lconnxr -lm
build:
	gcc $(CFLAGS) $(INCLUDE) -g -o0 -std=c99 -o test test.c $(LIBDIR) $(LIBS)
clean:
	rm -f *.o *.a *.gch
