#!/usr/bin/env python3
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MAIN_PATH = ROOT / "BlurFaces/main.py"
RUNTIME_PATH = ROOT / "BlurFaces/runtime.py"
REFLECTION_PATHS = (
    MAIN_PATH,
    RUNTIME_PATH,
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
        name = method_name(node.func.value)
        if name is not None:
            calls.setdefault(name, []).append(node)
            continue
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
                self.assertFalse(
                    node.keywords,
                    f"Reflection invoke in {path} uses keyword arguments: {ast.dump(node)}"
                )
                self.assertGreaterEqual(
                    len(node.args), 1,
                    f"invoke() must have at least receiver argument: {path}"
                )
                invocation_count += 1
        self.assertGreaterEqual(invocation_count, 4)

    def test_no_dynamic_proxy_in_main_plugin(self):
        main_source = MAIN_PATH.read_text(encoding="utf-8")
        self.assertNotIn("dynamic_proxy", main_source)


if __name__ == "__main__":
    unittest.main()
