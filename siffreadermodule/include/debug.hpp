#ifndef __DEBUG_HPP

#ifdef __DEBUG
    #define DEBUG(x) x
    #define DEBUG_IGNORE(x)
#else
    #define DEBUG(x)
    #define DEBUG_IGNORE(x) x
#endif // __DEBUG

#endif // __DEBUG_HPP