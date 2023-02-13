{
    extern "C++" {
        /* export all symbols in brt to enable external kernel registration */
        *brt::*;
    };
};
