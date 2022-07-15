#include "wrapper.h"
#include "detect.cpp.h"
#include "rcgn.cpp.h"

void rcgn_init_2() {
    rcgn_init();
}

void detect_init_2(unsigned x) {
    detect_init((void *)x);
}

int8_t *detect_get_output() {
    return detect_output(0)->data.int8;
}

int8_t *rcgn_get_output() {
    return rcgn_output(0)->data.int8;
}

int8_t *detect_get_input() {
    return detect_input(0)->data.int8;
}

int8_t *rcgn_get_input() {
    return rcgn_input(0)->data.int8;
}

void wrapper_detect_invoke() {
    detect_invoke();
}

void wrapper_rcgn_invoke() {
    rcgn_invoke();
}
