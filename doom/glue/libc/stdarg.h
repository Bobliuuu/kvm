#ifndef TL_STDARG_H
#define TL_STDARG_H
// clang 14's NVPTX backend cannot emit __builtin_va_arg; the vararg buffer is packed with natural alignment, so walk it by hand.
typedef __builtin_va_list va_list;
#define va_start(ap, last) __builtin_va_start(ap, last)
#define va_end(ap) ((void)0)
#define va_copy(d, s) ((d) = (s))
#define va_arg(ap, T) ({ unsigned long _p = ((unsigned long)(ap) + (__alignof__(T) - 1)) & ~(unsigned long)(__alignof__(T) - 1); (ap) = (va_list)(_p + sizeof(T)); *(T*)_p; })
#endif
