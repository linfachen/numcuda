//
// Created by spinors on 18-12-3.
//

#ifndef NUMCUDA_KERNEL_H
#define NUMCUDA_KERNEL_H

void fill_value(char *data,double value,int size,Dtype dtype,int thread_size=32);
void fill_value(char *data,long long value,int size,Dtype dtype,int thread_size=32);



#endif //NUMCUDA_KERNEL_H