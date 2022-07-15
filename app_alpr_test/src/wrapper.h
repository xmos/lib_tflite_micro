/** THis file wraps the NNs up in very plain C/XC */

#ifdef __cplusplus
extern "C" {
#endif
#include "stdint.h"

#ifdef __XC__
    #define UNSAFE unsafe
    #define CHANEND chanend
#else
    #define UNSAFE /**/
    #define CHANEND unsigned
#endif
    
int8_t * UNSAFE detect_get_output();
int8_t * UNSAFE rcgn_get_output();
int8_t * UNSAFE detect_get_input();
int8_t * UNSAFE rcgn_get_input();
void detect_rcgn_init(CHANEND x);
void wrapper_detect_invoke();
void wrapper_rcgn_invoke();
    
#ifdef __cplusplus
}
#endif
