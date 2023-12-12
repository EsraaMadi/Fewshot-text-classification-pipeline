import setuptools

setuptools.setup(name='text_categorizer',
                 version='0.1',
                 description='A package use model (Bertopic) to use in topic modeling tasks and categorizing text',
                 url='#',
                 author='Esraa Madi',
                 install_requires=['bertopic',
                                   'numba',
                                   'plotly',
                                   'sentence-transformers',
                                   'transformers',
                                   'umap-learn'],
                 author_email='',
                 packages=setuptools.find_packages(),
                 zip_safe=False)