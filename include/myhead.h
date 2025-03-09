#ifndef MYHEADH
#define MYHEADH
#define CEIL_DIV(x, y) ((x + y - 1)/ y)
#define CHECK(call) {\
    const cudaError error = call;\
    if(error != cudaSuccess) {\
        printf("ERROR: %s %d , " __FILE__, __LINE__);\
        printf("code %d, reason %s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

#endif