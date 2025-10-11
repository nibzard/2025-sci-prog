# Data Directory

This directory contains datasets and data files used in the Scientific Programming course.

## Structure

- `raw/` - Raw, unprocessed data files
- `processed/` - Cleaned and processed data ready for analysis
- `external/` - Data from external sources
- `temp/` - Temporary data files that can be regenerated

## Usage

Data files in this directory are mounted to `/workspaces/data` inside the dev container, making them accessible from within the containerized development environment.

## Guidelines

- Keep raw data files unchanged - store processed versions in the `processed/` subdirectory
- Include a `README.md` in each dataset subdirectory describing the source, format, and any processing steps
- Use appropriate file formats (CSV, JSON, Parquet) based on the data type and size
- Consider version control for small data files, but use Git LFS or external storage for large datasets