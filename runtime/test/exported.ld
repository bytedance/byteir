{
    extern "C++" {
        /* export all symbols in brt to enable external kernel registration */
        *brt::*;
    };
    llvm_orc_registerEHFrameSectionWrapper;
    llvm_orc_deregisterEHFrameSectionWrapper;
};
