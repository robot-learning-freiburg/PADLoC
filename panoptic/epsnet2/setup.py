from os import path, listdir

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def find_sources(root_dir):
    sources = []
    for file in listdir(root_dir):
        _, ext = path.splitext(file)
        if ext in [".cpp", ".cu"]:
            sources.append(path.join(root_dir, file))

    return sources


def make_extension(name, package):
    return CUDAExtension(
        name="{}.{}._backend".format(package, name),
        sources=find_sources(path.join("src", name)),
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["--expt-extended-lambda"],
        },
        include_dirs=["./include/"],
    )


here = path.abspath(path.dirname(__file__))

#with open(path.join(here, "README.md"), encoding="utf-8") as f:
#    long_description = f.read()

setuptools.setup(
    # Meta-data
    name="epsnet",
    author="Rohit Mohan",
    author_email="rohit.mohan@students.uni-freiburg.de",
    description="Efficient Panoptic Segmentation Network for Pytorch",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],

    # Versioning
    version='0.1.0',

    # Requirements
    setup_requires=["setuptools_scm"],
    python_requires=">=3, <4",

    # Package description
    packages=[
        "epsnet",
        "epsnet.algos",
        "epsnet.config",
        "epsnet.data",
        "epsnet.models",
        "epsnet.modules",
        "epsnet.modules.heads",
        "epsnet.utils",
        "epsnet.utils.bbx",
        "epsnet.utils.nms",
        "epsnet.utils.parallel",
        "epsnet.utils.roi_sampling",
    ],
    ext_modules=[
        make_extension("nms", "epsnet.utils"),
        make_extension("bbx", "epsnet.utils"),
        make_extension("roi_sampling", "epsnet.utils")
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
