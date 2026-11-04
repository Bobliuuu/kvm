// Minimal freestanding C library for running Doom on the GPU (compiled by clang for NVPTX).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>

int errno;
// ---------------- host-provided resources
const unsigned char* tl_wad; long tl_wad_len;
// the engine's own log (printf/puts/exit): a device-global ring-less buffer the host reads with cudaMemcpyFromSymbol
char doom_log[1 << 16]; volatile long doom_log_n; volatile int tl_exited;

static FILE s_out = {0, 0, 0}, s_err = {0, 0, 0}, s_in = {0, 0, 0};
FILE* tl_stdout = &s_out; FILE* tl_stderr = &s_err; FILE* tl_stdin = &s_in;
static FILE s_pool[4]; static int s_pool_used[4];

static void dbg_putc(char c) { if (doom_log_n < (long)sizeof doom_log - 1) { doom_log[doom_log_n] = c; doom_log_n = doom_log_n + 1; doom_log[doom_log_n] = 0; } }

// ---------------- memory
static char tl_heap[24u << 20] __attribute__((aligned(16)));
static unsigned long tl_hp;
static void tl_pool_reset(void) { for (int i = 0; i < 4; i++) s_pool_used[i] = 0; }
void tl_heap_reset(void) { tl_hp = 0; tl_pool_reset(); }
void* malloc(size_t n) {
  n = (n + 15) & ~15ul; if (tl_hp + n + 16 > sizeof tl_heap) return 0;
  char* p = tl_heap + tl_hp; *(unsigned long*)p = n; tl_hp += n + 16; return p + 16;
}
void* calloc(size_t a, size_t b) { void* p = malloc(a * b); if (p) memset(p, 0, a * b); return p; }
void free(void* p) { (void)p; }
void* realloc(void* p, size_t n) { if (!p) return malloc(n); size_t old = *(unsigned long*)((char*)p - 16); if (n <= old) return p; void* q = malloc(n); if (q) memcpy(q, p, old); return q; }


// ---------------- memory primitives (word fast path; LLVM never recurses into these because it skips loop-idiom rewrites inside functions named memcpy/memset)
void* memcpy(void* d, const void* s, size_t n) {
  unsigned char* dd = d; const unsigned char* ss = s;
  if ((((unsigned long)dd | (unsigned long)ss) & 7) == 0) { while (n >= 8) { *(unsigned long*)dd = *(const unsigned long*)ss; dd += 8; ss += 8; n -= 8; } }
  while (n--) *dd++ = *ss++;
  return d;
}
void* memmove(void* d, const void* s, size_t n) {
  unsigned char* dd = d; const unsigned char* ss = s;
  if (dd == ss || !n) return d;
  if (dd < ss || dd >= ss + n) return memcpy(d, s, n);
  while (n--) dd[n] = ss[n];
  return d;
}
void* memset(void* d, int c, size_t n) {
  unsigned char* dd = d; unsigned long w = (unsigned char)c; w |= w << 8; w |= w << 16; w |= w << 32;
  if (((unsigned long)dd & 7) == 0) { while (n >= 8) { *(unsigned long*)dd = w; dd += 8; n -= 8; } }
  while (n--) *dd++ = (unsigned char)c;
  return d;
}

// ---------------- strings
size_t strlen(const char* s) { size_t n = 0; while (s[n]) n++; return n; }
char* strcpy(char* d, const char* s) { char* r = d; while ((*d++ = *s++)); return r; }
char* strncpy(char* d, const char* s, size_t n) { size_t i = 0; for (; i < n && s[i]; i++) d[i] = s[i]; for (; i < n; i++) d[i] = 0; return d; }
char* strcat(char* d, const char* s) { strcpy(d + strlen(d), s); return d; }
char* strncat(char* d, const char* s, size_t n) { char* e = d + strlen(d); size_t i = 0; for (; i < n && s[i]; i++) e[i] = s[i]; e[i] = 0; return d; }
int strcmp(const char* a, const char* b) { while (*a && *a == *b) { a++; b++; } return (unsigned char)*a - (unsigned char)*b; }
int strncmp(const char* a, const char* b, size_t n) { for (size_t i = 0; i < n; i++) { if (a[i] != b[i]) return (unsigned char)a[i] - (unsigned char)b[i]; if (!a[i]) return 0; } return 0; }
int toupper(int c) { return (c >= 'a' && c <= 'z') ? c - 32 : c; }
int tolower(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }
int isspace(int c) { return c == ' ' || (c >= 9 && c <= 13); }
int isdigit(int c) { return c >= '0' && c <= '9'; }
int isalpha(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
int isalnum(int c) { return isalpha(c) || isdigit(c); }
int isupper(int c) { return c >= 'A' && c <= 'Z'; }
int islower(int c) { return c >= 'a' && c <= 'z'; }
int isprint(int c) { return c >= 32 && c < 127; }
int isxdigit(int c) { return isdigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
int ispunct(int c) { return isprint(c) && !isalnum(c) && c != ' '; }
int strcasecmp(const char* a, const char* b) { while (*a && tolower(*a) == tolower(*b)) { a++; b++; } return tolower((unsigned char)*a) - tolower((unsigned char)*b); }
int strncasecmp(const char* a, const char* b, size_t n) { for (size_t i = 0; i < n; i++) { int x = tolower((unsigned char)a[i]), y = tolower((unsigned char)b[i]); if (x != y) return x - y; if (!x) return 0; } return 0; }
int memcmp(const void* a, const void* b, size_t n) { const unsigned char *x = a, *y = b; for (size_t i = 0; i < n; i++) if (x[i] != y[i]) return x[i] - y[i]; return 0; }
void* memchr(const void* s, int c, size_t n) { const unsigned char* p = s; for (size_t i = 0; i < n; i++) if (p[i] == (unsigned char)c) return (void*)(p + i); return 0; }
char* strchr(const char* s, int c) { for (;; s++) { if (*s == (char)c) return (char*)s; if (!*s) return 0; } }
char* strrchr(const char* s, int c) { const char* r = 0; for (;; s++) { if (*s == (char)c) r = s; if (!*s) return (char*)r; } }
char* strstr(const char* h, const char* n) { size_t l = strlen(n); if (!l) return (char*)h; for (; *h; h++) if (!strncmp(h, n, l)) return (char*)h; return 0; }
char* strdup(const char* s) { size_t n = strlen(s) + 1; char* p = malloc(n); if (p) memcpy(p, s, n); return p; }
char* strerror(int e) { (void)e; return "error"; }

// ---------------- numbers
int abs(int x) { return x < 0 ? -x : x; }
long labs(long x) { return x < 0 ? -x : x; }
unsigned long strtoul(const char* s, char** end, int base) {
  while (isspace(*s)) s++; if (*s == '+') s++;
  if ((base == 0 || base == 16) && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) { s += 2; base = 16; } else if (base == 0) base = (s[0] == '0') ? 8 : 10;
  unsigned long v = 0;
  for (;; s++) { int d = isdigit(*s) ? *s - '0' : isalpha(*s) ? tolower(*s) - 'a' + 10 : 99; if (d >= base) break; v = v * base + d; }
  if (end) *end = (char*)s; return v;
}
long strtol(const char* s, char** end, int base) { while (isspace(*s)) s++; int neg = 0; if (*s == '-') { neg = 1; s++; } long v = (long)strtoul(s, end, base); return neg ? -v : v; }
int atoi(const char* s) { return (int)strtol(s, 0, 10); }
long atol(const char* s) { return strtol(s, 0, 10); }
double atof(const char* s) { while (isspace(*s)) s++; int neg = 0; if (*s == '-') { neg = 1; s++; } double v = 0; while (isdigit(*s)) v = v * 10 + (*s++ - '0'); if (*s == '.') { s++; double f = 0.1; while (isdigit(*s)) { v += (*s++ - '0') * f; f *= 0.1; } } return neg ? -v : v; }
static unsigned long rnd = 1;
int rand(void) { rnd = rnd * 6364136223846793005ul + 1442695040888963407ul; return (int)((rnd >> 33) & 32767); }
void srand(unsigned s) { rnd = s; }
void qsort(void* base, size_t n, size_t sz, int (*cmp)(const void*, const void*)) {
  char* b = base; char tmp[64];
  for (size_t i = 1; i < n; i++) { memcpy(tmp, b + i * sz, sz); size_t j = i; while (j > 0 && cmp(b + (j - 1) * sz, tmp) > 0) { memcpy(b + j * sz, b + (j - 1) * sz, sz); j--; } memcpy(b + j * sz, tmp, sz); }
}
char* getenv(const char* n) { (void)n; return 0; }
int system(const char* c) { (void)c; return -1; }
int atexit(void (*f)(void)) { (void)f; return 0; }
void exit(int code) { dbg_putc('\n'); const char* m = "[doom exit "; while (*m) dbg_putc(*m++); dbg_putc('0' + (code & 7)); dbg_putc(']'); tl_exited = 1; __builtin_trap(); }
void abort(void) { exit(134); }
int access(const char* p, int m) { (void)p; (void)m; return -1; }
int unlink(const char* p) { (void)p; return -1; }
int usleep(unsigned u) { (void)u; return 0; }
unsigned sleep(unsigned u) { (void)u; return 0; }
int mkdir(const char* p, int m) { (void)p; (void)m; return -1; }
int remove(const char* p) { (void)p; return -1; }
int rename(const char* a, const char* b) { (void)a; (void)b; return -1; }

// ---------------- printf family
typedef struct { char* buf; size_t cap, n; FILE* f; } sink_t;
static void sink_put(sink_t* s, char c) {
  if (s->f) { dbg_putc(c); s->n++; return; }
  if (s->n + 1 < s->cap) s->buf[s->n] = c; s->n++;
}
static void put_str(sink_t* s, const char* p, int len, int width, int left, char pad) {
  int padn = width > len ? width - len : 0;
  if (!left) while (padn-- > 0) sink_put(s, pad);
  for (int i = 0; i < len; i++) sink_put(s, p[i]);
  if (left) while (padn-- > 0) sink_put(s, ' ');
}
static int vfmt(sink_t* s, const char* fmt, va_list ap) {
  for (; *fmt; fmt++) {
    if (*fmt != '%') { sink_put(s, *fmt); continue; }
    fmt++; int left = 0, zero = 0, plus = 0, alt = 0, space = 0;
    for (;; fmt++) { if (*fmt == '-') left = 1; else if (*fmt == '0') zero = 1; else if (*fmt == '+') plus = 1; else if (*fmt == '#') alt = 1; else if (*fmt == ' ') space = 1; else break; }
    int width = 0, prec = -1;
    if (*fmt == '*') { width = va_arg(ap, int); if (width < 0) { left = 1; width = -width; } fmt++; } else while (isdigit(*fmt)) width = width * 10 + (*fmt++ - '0');
    if (*fmt == '.') { fmt++; prec = 0; if (*fmt == '*') { prec = va_arg(ap, int); fmt++; } else while (isdigit(*fmt)) prec = prec * 10 + (*fmt++ - '0'); }
    int lng = 0; while (*fmt == 'l' || *fmt == 'z' || *fmt == 't' || *fmt == 'j') { lng = 1; fmt++; } while (*fmt == 'h') fmt++;
    char c = *fmt; char tmp[72]; int tl = 0;
    if (c == 'd' || c == 'i' || c == 'u' || c == 'x' || c == 'X' || c == 'o' || c == 'p') {
      unsigned long v; int neg = 0; int base = (c == 'x' || c == 'X' || c == 'p') ? 16 : (c == 'o') ? 8 : 10;
      if (c == 'p') { v = (unsigned long)va_arg(ap, void*); alt = 1; }
      else if (c == 'd' || c == 'i') { long sv = lng ? va_arg(ap, long) : (long)va_arg(ap, int); if (sv < 0) { neg = 1; v = (unsigned long)(-sv); } else v = (unsigned long)sv; }
      else v = lng ? va_arg(ap, unsigned long) : (unsigned long)va_arg(ap, unsigned);
      const char* dig = (c == 'X') ? "0123456789ABCDEF" : "0123456789abcdef";
      do { tmp[tl++] = dig[v % base]; v /= base; } while (v && tl < 70);
      while (tl < prec && tl < 70) tmp[tl++] = '0';
      if (alt && base == 16) { tmp[tl++] = 'x'; tmp[tl++] = '0'; }
      if (neg) tmp[tl++] = '-'; else if (plus && (c == 'd' || c == 'i')) tmp[tl++] = '+'; else if (space && (c == 'd' || c == 'i')) tmp[tl++] = ' ';
      char rev[72]; for (int i = 0; i < tl; i++) rev[i] = tmp[tl - 1 - i];
      put_str(s, rev, tl, width, left, (zero && !left && prec < 0) ? '0' : ' ');
    } else if (c == 's') {
      const char* p = va_arg(ap, const char*); if (!p) p = "(null)"; int len = 0; while (p[len] && (prec < 0 || len < prec)) len++;
      put_str(s, p, len, width, left, ' ');
    } else if (c == 'c') { char ch = (char)va_arg(ap, int); put_str(s, &ch, 1, width, left, ' '); }
    else if (c == 'f' || c == 'g' || c == 'e' || c == 'F') { (void)va_arg(ap, double); put_str(s, "?", 1, width, left, ' '); }
    else if (c == '%') sink_put(s, '%');
    else if (!c) break;
  }
  return (int)s->n;
}
int vsnprintf(char* b, size_t n, const char* f, va_list ap) { sink_t s = {b, n, 0, 0}; int r = vfmt(&s, f, ap); if (n) b[s.n < n ? s.n : n - 1] = 0; return r; }
int snprintf(char* b, size_t n, const char* f, ...) { va_list ap; va_start(ap, f); int r = vsnprintf(b, n, f, ap); va_end(ap); return r; }
int sprintf(char* b, const char* f, ...) { va_list ap; va_start(ap, f); int r = vsnprintf(b, 1u << 20, f, ap); va_end(ap); return r; }
int vfprintf(FILE* fp, const char* f, va_list ap) { sink_t s = {0, 0, 0, fp ? fp : tl_stdout}; return vfmt(&s, f, ap); }
int vprintf(const char* f, va_list ap) { return vfprintf(tl_stdout, f, ap); }
int printf(const char* f, ...) { va_list ap; va_start(ap, f); int r = vfprintf(tl_stdout, f, ap); va_end(ap); return r; }
int fprintf(FILE* fp, const char* f, ...) { va_list ap; va_start(ap, f); int r = vfprintf(fp, f, ap); va_end(ap); return r; }
int puts(const char* s) { while (*s) dbg_putc(*s++); dbg_putc('\n'); return 0; }
int putchar(int c) { dbg_putc((char)c); return c; }
int fputc(int c, FILE* f) { (void)f; dbg_putc((char)c); return c; }
int putc(int c, FILE* f) { return fputc(c, f); }
int fputs(const char* s, FILE* f) { (void)f; while (*s) dbg_putc(*s++); return 0; }
int fflush(FILE* f) { (void)f; return 0; }

// ---------------- files: only the in-memory WAD can be opened (read only)
FILE* fopen(const char* name, const char* mode) {
  if (mode[0] != 'r' || !tl_wad) return 0;
  size_t n = strlen(name); if (n < 9 || strcasecmp(name + n - 9, "doom1.wad") != 0) return 0;
  for (int i = 0; i < 4; i++) if (!s_pool_used[i]) { s_pool_used[i] = 1; s_pool[i].kind = 1; s_pool[i].pos = 0; s_pool[i].err = 0; return &s_pool[i]; }
  return 0;
}
int fclose(FILE* f) { for (int i = 0; i < 4; i++) if (f == &s_pool[i]) s_pool_used[i] = 0; return 0; }
size_t fread(void* buf, size_t sz, size_t cnt, FILE* f) {
  if (!f || f->kind != 1 || !sz) return 0; long want = (long)(sz * cnt), have = tl_wad_len - f->pos; long n = want < have ? want : have; if (n <= 0) return 0;
  memcpy(buf, tl_wad + f->pos, (size_t)n); f->pos += n; return (size_t)n / sz;
}
size_t fwrite(const void* b, size_t sz, size_t cnt, FILE* f) { if (f && f->kind == 0) { const char* p = b; for (size_t i = 0; i < sz * cnt; i++) dbg_putc(p[i]); return cnt; } return 0; }
int fseek(FILE* f, long off, int wh) { if (!f || f->kind != 1) return -1; long np = wh == SEEK_SET ? off : wh == SEEK_CUR ? f->pos + off : tl_wad_len + off; if (np < 0) return -1; f->pos = np; return 0; }
long ftell(FILE* f) { return f ? f->pos : -1; }
int feof(FILE* f) { return f && f->kind == 1 && f->pos >= tl_wad_len; }
int ferror(FILE* f) { (void)f; return 0; }
int fgetc(FILE* f) { unsigned char c; return fread(&c, 1, 1, f) == 1 ? c : EOF; }
char* fgets(char* b, int n, FILE* f) { int i = 0; while (i < n - 1) { int c = fgetc(f); if (c == EOF) break; b[i++] = (char)c; if (c == '\n') break; } b[i] = 0; return i ? b : 0; }
int fscanf(FILE* f, const char* fmt, ...) { (void)f; (void)fmt; return EOF; }
int sscanf(const char* str, const char* fmt, ...) {
  va_list ap; va_start(ap, fmt); int cnt = 0;
  for (; *fmt; fmt++) {
    if (isspace(*fmt)) { while (isspace(*str)) str++; continue; }
    if (*fmt != '%') { if (*str != *fmt) break; str++; continue; }
    fmt++; int lng = 0; while (*fmt == 'l') { lng = 1; fmt++; }
    while (isspace(*str) && *fmt != 'c') str++;
    if (*fmt == 'd' || *fmt == 'i') { char* e; long v = strtol(str, &e, 10); if (e == str) break; str = e; void* dp = va_arg(ap, void*); if (lng) *(long*)dp = v; else *(int*)dp = (int)v; cnt++; }
    else if (*fmt == 'x') { char* e; unsigned long v = strtoul(str, &e, 16); if (e == str) break; str = e; void* dp = va_arg(ap, void*); *(unsigned*)dp = (unsigned)v; cnt++; }
    else if (*fmt == 's') { char* d = (char*)va_arg(ap, void*); if (!*str) break; while (*str && !isspace(*str)) *d++ = *str++; *d = 0; cnt++; }
    else if (*fmt == 'c') { if (!*str) break; *(char*)va_arg(ap, void*) = *str++; cnt++; }
    else break;
  }
  va_end(ap); return cnt;
}
