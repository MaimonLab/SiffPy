#ifndef __DEBUG_HPP

#ifdef __DEBUG
    #define DEBUG(...) __VA_ARGS__
    #define DEBUG_IGNORE(...)
#else
    #define DEBUG(...)
    #define DEBUG_IGNORE(...) __VA_ARGS__
#endif // __DEBUG

#endif // __DEBUG_HPP