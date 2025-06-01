# Plugin Packaging

Plugins are distributed as small directories that contain the compiled shared library and optional metadata. The helper program `plugin_packager` bundles and installs these directories so they can be discovered by `load_plugins_from_path()`.

## Directory layout

```
my_plugin/
  libmy_plugin.so
  plugin.json
```

`plugin.json` is optional and may list fields such as the plugin name, version and any library dependencies. Applications can inspect this file before loading the plugin.

## Creating an archive

Use the packaging mode to create a compressed archive of the plugin directory:

```bash
plugin_packager package my_plugin my_plugin.tar.gz
```

The command uses the system `tar` utility to write the archive.

## Installing a plugin

Extract the archive into a directory that is part of the plugin search path:

```bash
plugin_packager install my_plugin.tar.gz ~/.local/harmonics/plugins
export HARMONICS_PLUGIN_PATH=~/.local/harmonics/plugins
```

`load_plugins_from_path()` will scan the directory recursively and load any libraries that expose the `harmonics_register` entry point.

