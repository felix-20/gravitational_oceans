import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

PATH_TO_TEST_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test')
PATH_TO_TRAIN_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train')
PATH_TO_MODEL_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

# setup
if not os.path.isdir(PATH_TO_TRAIN_FOLDER):
    os.makedirs(PATH_TO_TRAIN_FOLDER)
if not os.path.isdir(PATH_TO_TEST_FOLDER):
    os.makedirs(PATH_TO_TEST_FOLDER)
if not os.path.isdir(PATH_TO_MODEL_FOLDER):
    os.makedirs(PATH_TO_MODEL_FOLDER)

def print_red(text: str):
    print(f'{bcolors.FAIL}{text}{bcolors.ENDC}')


def print_blue(text: str):
    print(f'{bcolors.OKCYAN}{text}{bcolors.ENDC}')


def print_green(text: str):
    print(f'{bcolors.OKGREEN}{text}{bcolors.ENDC}')

def print_yellow(text: str):
    print(f'{bcolors.WARNING}{text}{bcolors.ENDC}')
