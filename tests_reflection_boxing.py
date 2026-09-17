#!/usr/bin/env python3
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN_PATH = ROOT / "BlurFaces/main.py"
RUNTIME_PATH = ROOT / "BlurFaces/runtime.py"
BRIDGE_PATH = ROOT / "src/main/java/com/makey/blurfaces/g2/ModelSettingsBridge.java"
REFLECTION_PATHS = (
    MAIN_PATH,
    RUNTIME_PATH,
    ROOT / "loader/dex.py",
)


def reflection_invocations(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = {}

    def method_name(lookup):
        if (isinstance(lookup, ast.Call)
                and isinstance(lookup.func, ast.Attribute)
                and lookup.func.attr == "getMethod"
                and lookup.args
                and isinstance(lookup.args[0], ast.Constant)):
            return lookup.args[0].value
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "invoke":
            continue
        # Direct chain: getMethod(...).invoke(...)
        name = method_name(node.func.value)
        if name is not None:
            calls.setdefault(name, []).append(node)
            continue
        # Two-step: method = bridge.getMethod(...); method.invoke(...)
        lookup = node.func.value
        if isinstance(lookup, ast.Name):
            for assign in ast.walk(tree):
                if (isinstance(assign, ast.Assign)
                        and len(assign.targets) == 1
                        and isinstance(assign.targets[0], ast.Name)
                        and assign.targets[0].id == lookup.id):
                    name = method_name(assign.value)
                    if name is not None:
                        calls.setdefault(name, []).append(node)
                    break
    return calls


def is_boxed_java_integer(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Integer"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "jint"
        and len(node.args[0].args) == 1
        and not node.args[0].keywords
    )


class ReflectionBoxingTest(unittest.TestCase):
    def test_every_numeric_attach_and_update_argument_is_boxed_integer(self):
        calls = reflection_invocations(MAIN_PATH)
        for method_name in ("attach", "update"):
            with self.subTest(method=method_name):
                self.assertEqual(1, len(calls.get(method_name, [])))
                invocation = calls[method_name][0]
                self.assertEqual(14, len(invocation.args))
                self.assertTrue(is_boxed_java_integer(invocation.args[12]))
                self.assertTrue(is_boxed_java_integer(invocation.args[13]))

    def test_every_reflected_primitive_int_argument_is_boxed_integer(self):
        main_calls = reflection_invocations(MAIN_PATH)
        runtime_calls = reflection_invocations(RUNTIME_PATH)
        primitive_int_arguments = (
            main_calls["attach"][0].args[12:14]
            + main_calls["update"][0].args[12:14]
            + runtime_calls["getItem"][0].args[1:2]
            + runtime_calls["getPresetItem"][0].args[1:2]
        )
        self.assertEqual(6, len(primitive_int_arguments))
        self.assertTrue(all(is_boxed_java_integer(arg) for arg in primitive_int_arguments))

    def test_method_invoke_varargs_are_always_positional(self):
        invocation_count = 0
        for path in REFLECTION_PATHS:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "invoke"
                ):
                    continue
                invocation_count += 1
                self.assertTrue(node.args, path)
                self.assertIsInstance(node.args[0], ast.Constant, path)
                self.assertIsNone(node.args[0].value, path)
                self.assertFalse(node.keywords, path)
                self.assertFalse(
                    any(isinstance(arg, (ast.List, ast.Tuple, ast.Starred)) for arg in node.args[1:]),
                    path,
                )
        self.assertGreaterEqual(invocation_count, 20)

    def test_reflected_java_signatures_remain_primitive_int(self):
        main = MAIN_PATH.read_text(encoding="utf-8")
        bridge = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("IntConsumer, IntConsumer, String, String, String, String,", main)
        self.assertIn("Integer.TYPE, Integer.TYPE,", main)
        self.assertIn("CustomSetting.Factory<ModelRadioCell> getFactory(int index)", bridge)
        for signature in ("void attach(", "boolean update("):
            with self.subTest(signature=signature):
                start = bridge.index(signature)
                end = bridge.index(") {", start)
                declaration = bridge[start:end]
                self.assertIn("int selectedModel", declaration)
                self.assertIn("int downloadedMask", declaration)


if __name__ == "__main__":
    unittest.main()
