import os
import sys
import inspect

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_class_info(module_name, module_path):
    print(f"\n--- Module: {module_name} ({module_path}) ---")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module_name:
                print(f"\nClass: {name}")
                # Base classes
                bases = [base.__name__ for base in obj.__bases__]
                print(f"  Bases: {', '.join(bases)}")
                
                # Constructor signature
                try:
                    sig = inspect.signature(obj.__init__)
                    print(f"  Constructor: {name}{sig}")
                except:
                    pass
                    
                # Public methods
                methods = [m for m, _ in inspect.getmembers(obj, predicate=inspect.isfunction) if not m.startswith('_')]
                if methods:
                    print(f"  Methods: {', '.join(methods)}")
            elif inspect.isfunction(obj) and obj.__module__ == module_name:
                if not name.startswith('_'):
                    try:
                        sig = inspect.signature(obj)
                        print(f"\nFunction: {name}{sig}")
                    except:
                        print(f"\nFunction: {name}")
    except Exception as e:
        print(f"Error analyzing {module_name}: {e}")

def main():
    models_dir = os.path.join("src", "models")
    if not os.path.exists(models_dir):
        print(f"Error: {models_dir} not found.")
        return
        
    print("=== Project Model Class Structure Analysis ===")
    
    for file in os.listdir(models_dir):
        if file.endswith(".py") and file != "__init__.py":
            module_name = file[:-3]
            module_path = os.path.join(models_dir, file)
            get_class_info(module_name, module_path)

if __name__ == "__main__":
    main()
