#include <omp.h>

int cores_clamp(int value) {
    int procs = omp_get_num_procs();
    return (procs < value) ? procs : value;
}
