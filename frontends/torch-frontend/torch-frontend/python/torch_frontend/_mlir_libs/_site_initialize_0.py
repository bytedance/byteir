def context_init_hook(context):
    from ._stablehlo import register_dialect as register_stablehlo_dialect
    from ._torchMlir import register_dialect as register_torch_dialect

    register_stablehlo_dialect(context)
    register_torch_dialect(context)
