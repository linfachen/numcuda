//
// Created by spinors on 18-12-3.
//

#ifndef NUMCUDA_KERNEL_H
#define NUMCUDA_KERNEL_H

void fill_value(char *data,double value,int size,Dtype dtype,int thread_size=32);
void fill_value(char *data,long long value,int size,Dtype dtype,int thread_size=32);
void elt_add_op(char *a,char *b,char *c,int size,Dtype dtype,int thread_size=32);
void elt_sub_op(char *a,char *b,char *c,int size,Dtype dtype,int thread_size=32);
void elt_mul_op(char *a,char *b,char *c,int size,Dtype dtype,int thread_size=32);
void elt_div_op(char *a,char *b,char *c,int size,Dtype dtype,int thread_size=32);

#endif //NUMCUDA_KERNEL_H
