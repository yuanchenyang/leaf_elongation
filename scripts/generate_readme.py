import re
import subprocess

# Pattern to match <<example.py>>
pattern = r"<<(.*?)>>"

template = \
"""```shell
$ python {} --help
{}
```
"""

def replace_with_help_string(match):
    script_name = match.group(1)

    # Run `python example.py -h` and capture the output
    try:
        help_output = subprocess.check_output(['python', script_name, '--help'], stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e.output}")
        return f"<<{script_name}>>"  # Keep the original if an error occurs

    # Return the help string
    return template.format(script_name, help_output.strip())

def process_file(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        content = file.read()

    # Replace <<example.py>> with the help string
    new_content = re.sub(pattern, replace_with_help_string, content)

    # Write the modified content back to the file
    with open(output_filename, 'w') as file:
        file.write(new_content)

if __name__ == "__main__":
    process_file('readme.template', 'readme.md')
