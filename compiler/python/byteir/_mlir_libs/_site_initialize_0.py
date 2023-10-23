def context_init_hook(context):
    from ._byteir import register_dialect_extensions, register_cat_dialect, register_translation_dialects
    from ._byteir import register_stablehlo_dialect, register_chlo_dialect
    from ._mlirHlo import register_mhlo_dialect

    register_stablehlo_dialect(context)
    register_chlo_dialect(context)
    register_cat_dialect(context)
    register_dialect_extensions(context)
    register_mhlo_dialect(context)
    register_translation_dialects(context)

    context.allow_unregistered_dialects = True
