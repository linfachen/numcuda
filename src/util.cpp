//
// Created by spinors on 18-12-4.
//


#include "util.h"

#define PAIR_NC_NP(a,b) std::pair<Dtype,NPY_TYPES>(a,b)
#define PAIR_NP_NC(a,b) std::pair<NPY_TYPES,Dtype>(a,b)
#define PAIR_NC_NPC(a,b) std::pair<Dtype,NPY_TYPECHAR>(a,b)
#define PAIR_NPC_NC(a,b) std::pair<NPY_TYPECHAR,Dtype>(a,b)

//from numcuda type to numpy type
std::unordered_map<Dtype,NPY_TYPES>   TO_NUMPY_TYPE={
        PAIR_NC_NP(Dtype::int8,NPY_BYTE),
        PAIR_NC_NP(Dtype::uint8,NPY_UBYTE),
        PAIR_NC_NP(Dtype::int16,NPY_SHORT),
        PAIR_NC_NP(Dtype::uint16,NPY_USHORT),
        PAIR_NC_NP(Dtype::int32,NPY_INT),
        PAIR_NC_NP(Dtype::uint32,NPY_UINT),
        PAIR_NC_NP(Dtype::int64,NPY_LONG),
        PAIR_NC_NP(Dtype::uint64,NPY_ULONG),
        PAIR_NC_NP(Dtype::float16,NPY_HALF),
        PAIR_NC_NP(Dtype::float32,NPY_FLOAT),
        PAIR_NC_NP(Dtype::float64,NPY_DOUBLE),
};


//from numcuda type to numpy char type
std::unordered_map<Dtype,NPY_TYPECHAR>   TO_NUMPY_CHARTYPE={
        PAIR_NC_NPC(Dtype::int8,NPY_BYTELTR),
        PAIR_NC_NPC(Dtype::uint8,NPY_UBYTELTR),
        PAIR_NC_NPC(Dtype::int16,NPY_SHORTLTR),
        PAIR_NC_NPC(Dtype::uint16,NPY_USHORTLTR),
        PAIR_NC_NPC(Dtype::int32,NPY_INTLTR),
        PAIR_NC_NPC(Dtype::uint32,NPY_UINTLTR),
        PAIR_NC_NPC(Dtype::int64,NPY_LONGLTR),
        PAIR_NC_NPC(Dtype::uint64,NPY_ULONGLTR),
        PAIR_NC_NPC(Dtype::float16,NPY_HALFLTR),
        PAIR_NC_NPC(Dtype::float32,NPY_FLOATLTR),
        PAIR_NC_NPC(Dtype::float64,NPY_DOUBLELTR),
};


//from numpy type to numcuda type
std::unordered_map<NPY_TYPES,Dtype>   TO_NUMCUDA_TYPE={
        PAIR_NP_NC(NPY_BYTE,Dtype::int8),
        PAIR_NP_NC(NPY_UBYTE,Dtype::uint8),
        PAIR_NP_NC(NPY_SHORT,Dtype::int16),
        PAIR_NP_NC(NPY_USHORT,Dtype::uint16),
        PAIR_NP_NC(NPY_INT,Dtype::int32),
        PAIR_NP_NC(NPY_UINT,Dtype::uint32),
        PAIR_NP_NC(NPY_LONG,Dtype::int64),
        PAIR_NP_NC(NPY_ULONG,Dtype::uint64),
        PAIR_NP_NC(NPY_HALF,Dtype::float16),
        PAIR_NP_NC(NPY_FLOAT,Dtype::float32),
        PAIR_NP_NC(NPY_DOUBLE,Dtype::float64),
};


//numcuda type to from numpy type
std::unordered_map<NPY_TYPECHAR,Dtype>   NUMPY_CHARTYPE_TO_NUMCUDA_TYPE={
        PAIR_NPC_NC(NPY_BYTELTR,Dtype::int8),
        PAIR_NPC_NC(NPY_UBYTELTR,Dtype::uint8),
        PAIR_NPC_NC(NPY_SHORTLTR,Dtype::int16),
        PAIR_NPC_NC(NPY_USHORTLTR,Dtype::uint16),
        PAIR_NPC_NC(NPY_INTLTR,Dtype::int32),
        PAIR_NPC_NC(NPY_UINTLTR,Dtype::uint32),
        PAIR_NPC_NC(NPY_LONGLTR,Dtype::int64),
        PAIR_NPC_NC(NPY_ULONGLTR,Dtype::uint64),
        PAIR_NPC_NC(NPY_HALFLTR,Dtype::float16),
        PAIR_NPC_NC(NPY_FLOATLTR,Dtype::float32),
        PAIR_NPC_NC(NPY_DOUBLELTR,Dtype::float64),
};
