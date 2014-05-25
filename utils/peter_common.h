#ifndef PETER_COMMON
#define PETER_COMMON

#include <stdio.h>

  
struct Logger {
    char *_log_name;
    FILE *_f;
    Logger(const char *log_name);
    ~Logger(); 
    void write_log(const char *file, int line, const char *format, ...);
};

#ifdef PETER_DEBUG
extern Logger *logger;
#define log_debug(...) logger->write_log(__FILE__, __LINE__, __VA_ARGS__)
#else
#define log_debug(...)
#endif 

#endif // #ifndef PETER_COMMON
