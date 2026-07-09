import os
import ast

def check_syntax(directory):
    errors = []
    for root, dirs, files in os.walk(directory):
        if '.venv' in dirs:
            dirs.remove('.venv')
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        source = f.read()
                    ast.parse(source, filename=filepath)
                except SyntaxError as e:
                    errors.append(f"Syntax Error in {filepath}:\nLine {e.lineno}: {e.msg}\n{e.text}")
                except Exception as e:
                    errors.append(f"Other Error in {filepath}: {e}")
    return errors

if __name__ == "__main__":
    errors = check_syntax('.')
    if errors:
        for e in errors:
            print(e)
            print("-" * 40)
    else:
        print("No syntax errors found.")
