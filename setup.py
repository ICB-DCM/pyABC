import setuptools

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except (IOError, ImportError):
    long_description = open('README.md').read()

setuptools.setup(
    long_description = long_description,
    long_description_content_type="text/x-rst"
)
