load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_cc_binary", "apollo_package", "apollo_component")
load("//tools:cpplint.bzl", "cpplint")
 
package(default_visibility = ["//visibility:public"])
 
apollo_cc_binary(
    name = "radar",
    srcs = ["radar.cc"],
    deps = [
        "//cyber",
        "//ZL2/proto:ZL2_proto",
    ],
    linkstatic = True,
)

apollo_package()
 
cpplint()
