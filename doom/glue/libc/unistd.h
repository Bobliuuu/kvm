#ifndef TL_UNISTD_H
#define TL_UNISTD_H
#include <stddef.h>
int access(const char*, int); int unlink(const char*); int usleep(unsigned); unsigned sleep(unsigned);
#define F_OK 0
#endif
