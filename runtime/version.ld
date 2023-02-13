{
global:
    extern "C++" {
        /* since we didn't have pubapi or capi yet, export all of symbols defined in brt namespace */
        *brt::*;
    };

local:
    extern "C++" {
        /* hide all of symbols defined in MLIR to avoid symbol conflict */
        *mlir::*;
    };
};
