# Notebooks Directory

This directory contains Jupyter notebooks, Marimo notebooks, and other interactive computational documents for the Scientific Programming course.

## Structure

- `tutorials/` - Step-by-step tutorials and learning materials
- `experiments/` - Experimental code and data exploration
- `assignments/` - Completed assignment notebooks
- `demos/` - Demonstration notebooks for course concepts
- `templates/` - Notebook templates for common tasks

## Usage

Notebooks in this directory are mounted to `/workspaces/notebooks` inside the dev container and can be accessed via:

- Jupyter Lab (port 8888)
- Jupyter Notebook (port 8888)
- Marimo (port 2818)
- VS Code's built-in notebook support

## Guidelines

- Use clear, descriptive filenames that indicate the topic and purpose
- Include markdown cells explaining the purpose, methodology, and conclusions
- Ensure notebooks can be executed from top to bottom without errors
- Keep computational cells focused and reasonably sized
- Use relative paths to access data files from the `/workspaces/data` directory
- Consider creating both analysis and presentation versions of complex notebooks

## Supported Formats

- `.ipynb` - Jupyter notebooks
- `.py` - Marimo notebooks (Python files with marimo UI components)
- `.md` - Markdown notebooks for documentation