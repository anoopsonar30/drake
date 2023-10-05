# This file marks a workspace root for the Bazel build system.
# See `https://bazel.build/`.

workspace(name = "drake")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//tools/workspace:default.bzl", "add_default_workspace")
load(
    "@drake//tools/workspace:pkg_config.bzl",
    "pkg_config_repository",
)

add_default_workspace()

# TODO(jwnimmer-tri) Before we enable Clarabel by default, we need to figure
# out how to provide the Rust toolchains as part of `default.bzl`. For now,
# we'll work around it by adding the toolchain definition to our WORKSPACE.
load("@rules_rust//rust:repositories.bzl", "rust_register_toolchains")

rust_register_toolchains()

# Add some special heuristic logic for using CLion with Drake.
load("//tools/clion:repository.bzl", "drake_clion_environment")

drake_clion_environment()

load("@bazel_skylib//lib:versions.bzl", "versions")

pkg_config_repository(
    name = "bullet",
    # licenses = ["ignore"],
    modname = "bullet",
)

# This needs to be in WORKSPACE or a repository rule for native.bazel_version
# to actually be defined. The minimum_bazel_version value should match the
# version passed to the find_package(Bazel) call in the root CMakeLists.txt.
<<<<<<< HEAD
versions.check(minimum_bazel_version = "4.0")
=======
versions.check(minimum_bazel_version = "5.1")

# The cargo_universe programs are only used by Drake's new_release tooling, not
# by any compilation rules. As such, we can put it directly into the WORKSPACE
# instead of into our `//tools/workspace:default.bzl` repositories.
load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")  # noqa

crate_universe_dependencies(bootstrap = True)
>>>>>>> 79b2df83452e18287fd32b75d4af77d0172a9c87
