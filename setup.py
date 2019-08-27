#!/usr/bin/env python
# -*- coding: utf-8 -*-
import atexit
import io
import json
import sys
from os import path, walk, makedirs
from shutil import copy

from setuptools import setup, find_packages
from setuptools.command.install import install


def _post_install():
    from subprocess import call
    call([sys.executable, '-m', 'nltk.downloader', 'punkt', 'averaged_perceptron_tagger', 'wordnet'])


class CustomInstaller(install):
    def __init__(self, *args, **kwargs):
        super(CustomInstaller, self).__init__(*args, **kwargs)
        atexit.register(_post_install)


def load_meta(fp):
    with io.open(fp, encoding='utf8') as f:
        return json.load(f)


def list_files(data_dir):
    output = []
    for root, _, filenames in walk(data_dir):
        for filename in filenames:
            if not filename.startswith('.'):
                output.append(path.join(root, filename))
    output = [path.relpath(p, path.dirname(data_dir)) for p in output]
    output.append('meta.json')
    return output


def setup_package():
    root = path.abspath(path.dirname(__file__))
    meta_path = path.join(root, 'meta.json')
    meta = load_meta(meta_path)
    model_name = str(meta['name'])
    model_dir = path.join(model_name, model_name + '-' + meta['version'])

    makedirs(model_dir, exist_ok=True)
    copy(meta_path, path.join(model_name))
    copy(meta_path, model_dir)

    setup(
        name=model_name,
        version=meta['version'],
        description=meta['description'],
        author=meta['author'],
        author_email=meta['email'],
        url=meta['url'],
        license=meta['license'],
        keywords=meta['keywords'],
        packages=find_packages(),
        package_data={model_name: list_files(model_dir)},
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Build Tools',
            'License ::  MIT License',
            'Programming Language :: Python :: 3.6',
        ],

        install_requires=['matplotlib', 'numpy', 'scipy==1.2.1', 'wordcloud', 'pandas', 'tqdm', 'nltk', 'scikit-learn',
                          'xlrd', 'python-Levenshtein', 'gensim==3.4.0', 'statsmodels', 'keras', 'tensorflow',
                          'keras_tqdm', 'patsy', 'humanfriendly', 'psutil', 'jinja2', 'urllib3==1.22'],
        # extras_require={'dev': ['check-manifest'],'test': ['coverage'],},
        python_requires='>=3.6',
        cmdclass={
            'install': CustomInstaller,
        },
    )


if __name__ == '__main__':
    setup_package()
