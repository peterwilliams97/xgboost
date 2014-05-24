#ifndef XGBOOST_UTILS_H
#define XGBOOST_UTILS_H
/*!
 * \file xgboost_utils.h
 * \brief simple utils to support the code
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */

#undef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#ifdef _MSC_VER
#define fopen64 fopen
#else

// use 64 bit offset, either to include this header in the beginning, or 
#ifdef _FILE_OFFSET_BITS
#if _FILE_OFFSET_BITS == 32
#warning "FILE OFFSET BITS defined to be 32 bit"
#endif
#endif

#ifdef __APPLE__
#define off64_t off_t
#define fopen64 fopen
#endif

#define _FILE_OFFSET_BITS 64
extern "C"{    
#include <sys/types.h>
};
#include <cstdio>
#endif

#include <cstdio>
#include <cstdlib>
#include <stdarg.h>
#include "peter_common.h"

namespace xgboost{
    /*! \brief namespace for helper utils of the project */
    namespace utils{

        inline void BaseWriter(const char *level, const char *format, va_list ap) {
            fprintf(stderr, "%s:", level);
            vfprintf(stderr, format, ap);
            fprintf(stderr, "\n");
            fflush(stderr);
        }

        inline void Error(const char *format, ...) {
            va_list ap;
            va_start(ap, format);
            BaseWriter("Error", format, ap);
            va_end(ap);
            exit(-1);
        }

        inline void Assert(bool exp){
            if (!exp) { 
                Error("AssertError");
            }
        }

        inline void Assert(bool exp, const char *format, ...) {
            if (!exp) {
                va_list ap;
                va_start(ap, format);
                BaseWriter("Error", format, ap);
                va_end(ap);
                exit(-1);
            }
        }

        inline void Warning(const char *format, ...){
             va_list ap;
             va_start(ap, format);
             BaseWriter("warning", format, ap);
             va_end(ap);
        }

        /*! \brief replace fopen, report error when the file open fails */
        inline FILE *FopenCheck(const char *fname, const char *flag){
            FILE *fp = fopen64(fname, flag);
            if (fp == NULL){
                fprintf(stderr, "can not open file \"%s\" \n", fname);
                fflush(stderr);
                exit(-1);
            }
            return fp;
        }
    };
};

#endif
