#ifndef TL_STDIO_H
#define TL_STDIO_H
#include <stddef.h>
#include <stdarg.h>
typedef struct _TLFILE { int kind; long pos; int err; } FILE;   // kind 0 = stdout/stderr sink, 1 = in-memory WAD (read only)
extern FILE* tl_stdout; extern FILE* tl_stderr; extern FILE* tl_stdin;
#define stdout tl_stdout
#define stderr tl_stderr
#define stdin tl_stdin
#define EOF (-1)
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#define BUFSIZ 512
int printf(const char*, ...); int fprintf(FILE*, const char*, ...); int sprintf(char*, const char*, ...);
int snprintf(char*, size_t, const char*, ...); int vsnprintf(char*, size_t, const char*, va_list); int vfprintf(FILE*, const char*, va_list); int vprintf(const char*, va_list);
int puts(const char*); int putchar(int); int fputs(const char*, FILE*); int fputc(int, FILE*); int putc(int, FILE*); int fflush(FILE*);
FILE* fopen(const char*, const char*); int fclose(FILE*); size_t fread(void*, size_t, size_t, FILE*); size_t fwrite(const void*, size_t, size_t, FILE*);
int fseek(FILE*, long, int); long ftell(FILE*); int feof(FILE*); int ferror(FILE*); int fgetc(FILE*); char* fgets(char*, int, FILE*);
int sscanf(const char*, const char*, ...); int fscanf(FILE*, const char*, ...); int remove(const char*); int rename(const char*, const char*);
#endif
