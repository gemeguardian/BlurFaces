# Native ELF plugin template

This is a minimal exteraGram native plugin template for `arm64-v8a`. The C++ library does nothing except log when Android loads or unloads it. The final `.plugin` contains the compiled ELF `.so` embedded as Base64, so only one file is installed on the phone.

## Build

```bash
cd /home/PluginDev/template-so
./build-native.sh
```

This runs the Android NDK build and then creates `build/plugin/myplugin.plugin`.

## Install

Install only:

```text
build/plugin/myplugin.plugin
```

The Python loader decodes the embedded `libmyplugin.so` into the app's private cache directory and calls `System.load()` on it. No separate `.so` push is required.

## Files

- `src/main.cpp`: native C++ entry points `JNI_OnLoad` and `JNI_OnUnload`.
- `Android.mk`, `Application.mk`: Android NDK build configuration.
- `build-native.sh`: builds the ELF and embeds it into the plugin.
- `loader/`: metadata, loader, and Base64 packer.

## Limitations

This template targets `arm64-v8a`, which is the architecture used by modern phones such as OnePlus 13. It does not provide a Java/Xposed bridge and it does not hook Telegram. Native code that needs Java interaction must add JNI calls and obtain a `JNIEnv*` from the `JavaVM*` received in `JNI_OnLoad`.
