#ifndef RUN_H
#define RUN_H

#ifndef __XC__
#define UNSAFE /**/
#else
#define UNSAFE unsafe
#endif

#ifdef __cplusplus
 class dummy_class {
    int k;
 };
 struct dummy_struct_t {
  dummy_class k;
 };

extern "C" {
#endif

struct dummy_struct_t;

 typedef struct dummy{
    struct dummy_struct_t *UNSAFE ptr ;
 } dummy_t;

#ifdef __XC__
    int run_detect(dummy_t * unsafe d, chanend ?c_flash);
    int run_rcgn();
#else
    int run_detect(dummy_t * d, void *flash_data);
    int run_rcgn();
#endif


#ifdef __cplusplus
    };
#endif

#endif
