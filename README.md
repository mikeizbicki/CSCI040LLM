# CSCI040LLM

[![Run Tests](https://github.com/NicoLS18/CSCI040LLM/actions/workflows/tests.yml/badge.svg)](https://github.com/NicoLS18/CSCI040LLM/actions/workflows/tests.yml)
[![Integration Test](https://github.com/NicoLS18/CSCI040LLM/actions/workflows/integration.yml/badge.svg)](https://github.com/NicoLS18/CSCI040LLM/actions/workflows/integration.yml)
[![flake8](https://github.com/NicoLS18/CSCI040LLM/actions/workflows/flake8.yml/badge.svg)](https://github.com/NicoLS18/CSCI040LLM/actions/workflows/flake8.yml)

A pirate-themed document chat agent powered by the Groq LLM API. Chat naturally with files in your project using tools to list directories, read files, search with regex, and perform calculations.

PyPI: https://pypi.org/project/cmc-csci005-nicolaslaub/

## Demo Video

https://youtu.be/Mu8IV-zt29s

https://github.com/NicoLS18/CSCI040LLM/blob/main/TTSVideo.mov  

## Usage Examples

### eBay Scraper (`test_projects/ebayscraper`)

```
chat> what files are in the ebay scraper project?
Arrr, the ebay scraper project be havin' these files: ebay-dl.py, lego.csv, lego.json, nike shoes.csv, nike shoes.json, README.md, vintage watch.csv, vintage watch.json!

chat> /cat test_projects/ebayscraper/README.md
# ebayscraper
...
```

### Custom Webpage (`test_projects/webpage`)

```
chat> what html files does the webpage project have?
Arrr, the webpage project be havin' these HTML files: animals.html, index.html, quiz1.html, visit.html, matey!

chat> /grep <title> test_projects/webpage/index.html
<title>Zoo</title>
```

### Markdown Compiler (`test_projects/markdown-compiler`)

```
chat> what does the markdown compiler's pyproject.toml say?
Arrr, the pyproject.toml be definin' the project name, version, dependencies, and entry point for the markdown compiler package, matey!

chat> /ls test_projects/markdown-compiler/markdown_compiler
```
