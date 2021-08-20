#include <stdio.h>
#include <csptr/smart_ptr.h>

void cleanup(void *ptr, void *meta){
    (void) meta;
    printf("fuck u");
}

int main(void) {
    // some_int is an unique_ptr to an int with a value of 1.
    smart int *some_int = unique_ptr(int, 1, cleanup);
    int *p = some_int;
    printf("%p = %d\n", some_int, *some_int);

    // some_int is destroyed here
    return 0;
}

