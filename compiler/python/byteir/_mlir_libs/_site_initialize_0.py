def context_init_hook(context):
    from ._byteir import register_cat_dialect, register_ace_dialect, register_ccl_dialect, register_byre_dialect, register_byre_serial_dialect
    from ._byteir import register_dialect_extensions, register_translation_dialects
    from ._mlirHlo import register_mhlo_dialect
    from ._stablehlo import register_dialect as register_stablehlo_dialect

    register_stablehlo_dialect(context)
    register_mhlo_dialect(context)
    register_cat_dialect(context)
    register_ace_dialect(context)
    register_ccl_dialect(context)
    register_byre_dialect(context)
    register_byre_serial_dialect(context)
    register_dialect_extensions(context)
    register_translation_dialects(context)

    context.allow_unregistered_dialects = True
