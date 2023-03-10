#!/usr/bin/env bash

python3 -m venv env
source ./env/bin/activate
python3 -m pip install -r ./requirements.txt
CWD=$(pwd)
export PYTHONPATH=$CWD
echo -e "\033[92mSignal generation\033[0m"
python3 ./src/data_management/generation/signal_generator.py
echo -e "\033[92mStatic noise generation\033[0m"
python3 ./src/data_management/generation/static_noise_generator.py
echo -e "\033[92mDynamic noise generation\033[0m"
python3 ./src/data_management/generation/dynamic_noise_generator.py
echo -e "\033[92mYou are set up and ready to surf gravitaional oceans. Stay safe!\033[0m"
