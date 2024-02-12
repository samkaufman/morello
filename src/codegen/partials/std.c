#include <inttypes.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>

#ifdef BYTE_ORDER
#if BYTE_ORDER == BIG_ENDIAN
#define LE_TO_CPU32(val) (((val & 0x000000FFU) << 24) | \
                          ((val & 0x0000FF00U) << 8) |  \
                          ((val & 0x00FF0000U) >> 8) |  \
                          ((val & 0xFF000000U) >> 24))
#define LE_TO_CPU16(val) (((val & 0x00FFU) << 8) | \
                          ((val & 0xFF00U) >> 8))
#else
#define LE_TO_CPU32(val) (val)
#define LE_TO_CPU16(val) (val)
#endif
#else
#error "BYTE_ORDER is not defined"
#endif
