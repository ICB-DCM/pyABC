import setuptools

try:
    import pypandoc
    content = pypandoc.convert_file('README.md', 'rst')
except (ImportError, OSError):
    with open('README.md', encoding='utf-8') as f:
        content = f.read()

with open('README.rst', 'w', encoding='utf-8') as f:
    f.write(content)

setuptools.setup()
