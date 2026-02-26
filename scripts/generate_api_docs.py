#!/usr/bin/env python3
"""Generate API documentation."""

import inspect
import pkgutil
import importlib
from pathlib import Path
import aerotica


def generate_docs(output_dir: Path):
    """Generate API documentation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    modules = []
    
    # Find all submodules
    for _, name, _ in pkgutil.walk_packages(
        aerotica.__path__, aerotica.__name__ + '.'
    ):
        try:
            module = importlib.import_module(name)
            modules.append((name, module))
        except ImportError as e:
            print(f"Could not import {name}: {e}")
    
    # Generate documentation
    for module_name, module in modules:
        doc_file = output_dir / f"{module_name.replace('.', '_')}.md"
        
        with open(doc_file, 'w') as f:
            f.write(f"# {module_name}\n\n")
            
            if module.__doc__:
                f.write(module.__doc__ + "\n\n")
            
            # Find classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module_name:
                    f.write(f"## {name}\n\n")
                    if obj.__doc__:
                        f.write(obj.__doc__ + "\n\n")
                    
                    # Find methods
                    for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                        if not method_name.startswith('_'):
                            f.write(f"### {method_name}\n\n")
                            if method.__doc__:
                                f.write(method.__doc__ + "\n\n")
                            
                            # Show signature
                            sig = inspect.signature(method)
                            f.write(f"```python\n{method_name}{sig}\n```\n\n")
    
    print(f"✅ Generated {len(modules)} module docs in {output_dir}")


if __name__ == "__main__":
    generate_docs(Path("docs/api"))
