import os
from conans import ConanFile, tools, CMake


class apsw_mod_trajectory_planningConan(ConanFile):
    name = "apsw_mod_trajectory_planning"
    version = "1.6.0"
    _framework_version = "1.6.0"
    description = "Trajektorienplanung APSW-Projekt"
    settings = "os", "compiler", "build_type", "arch"
    scm = {"type": "git", "url": "auto", "revision": "auto"}
    generators = "cmake"
    no_copy_source = True
    short_paths = True
    options = {"docs": [True, False],
               "logger": ["rte", "local", "file", "none"],
               "profiling": [True, False],
               "optimizing": [True, False]}
    default_options = {"docs": True,
                       "logger": "file",
                       "profiling": False,
                       "optimizing": True}

    def build_requirements(self):
        self.build_requires("sep_framework/%s@%s/%s" %
                            (self._framework_version, self.user, "stable"))
        self.build_requires("gtest/%s@%s/%s" % ("1.8.1", self.user, "stable"))

    def build(self):
        cmake = CMake(self)
        cmake.definitions["CMAKE_CONFIGURATION_TYPES"] = self.settings.build_type

        if self.options.docs:
            cmake.definitions["DOCS"] = "ON"

        if self.options.logger == "rte":
            cmake.definitions["RTE_LOGGER"] = "ON"
        elif self.options.logger == "local":
            cmake.definitions["LOCAL_LOGGER"] = "ON"
        elif self.options.logger == "file":
            cmake.definitions["FILE_LOGGER"] = "ON"

        if self.settings.os == "Windows":
            cmake.definitions["LOCAL_LOGGER_WINDOWS"] = "ON"
        else:
            cmake.definitions["LOCAL_LOGGER_LINUX"] = "ON"

        if self.options.profiling:
            cmake.definitions["PROFILING"] = "ON"

        if self.options.optimizing == True:
            cmake.definitions["OPTIMIZATION"] = "ON"
        else:
            cmake.definitions["OPTIMIZATION"] = "OFF"

        if self.options.docs:
            cmake.definitions["DOCS"] = "ON"

        cmake.configure()
        cmake.build()
        cmake.install()
        cmake.test()

    def package_id(self):
        del self.info.settings.compiler

    def package_info(self):
        self.user_info.sep_install_files = "TP.rpp"
        self.user_info.sep_settings_file = None
