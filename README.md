# CALE

## Description

CALE, short for *Cooperative and Adversarial LEarning*, is an open-source implementation of [Cooperative and Adversarial Learning: Co-Enhancing Discriminability and Transferability in Domain Adaptation]().

## Installation

To use CALE, you need to install Transfer Learning Library [dalib](https://github.com/thuml/Transfer-Learning-Library).

```lang: sh
git clone https://github.com/thuml/Transfer-Learning-Library.git
cd Transfer-Learning-Library/
python setup.py install
```

To install other dependent packages via pip:

```lang: sh
pip install -r requirements.txt
```

## Documentation

To train on Office-Home dataset, run scripts

```lang: sh
% cd PROJECT_DIR
bash scripts/run_on_officehome.sh
````

## License

[MIT](https://choosealicense.com/licenses/mit/)
