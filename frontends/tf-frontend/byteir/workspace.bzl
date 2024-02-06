def ace_repo():
    native.new_local_repository(
        name = "byteir",
        path = "./../../compiler/dialects",
        build_file = "//byteir:ace.BUILD",
    )
