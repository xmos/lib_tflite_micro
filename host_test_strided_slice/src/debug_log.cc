#include <cstdio>

extern "C" void DebugLog(const char* s) { while (*s) { putchar(*s); s++; }}