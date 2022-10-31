import os

from kaggle.api.kaggle_api_extended import KaggleApi

from data_generator import GODataGenerator

if not os.path.isdir('./data'):
    os.makedirs('./data')

# api = KaggleApi()
# api.authenticate()

# # download single file
# # Signature: dataset_download_file(dataset, file_name, path=None, force=False, quiet=True)
# all_files = api.competitions_data_list_files('g2net-detecting-continuous-gravitational-waves')
# with open('./data/all_files.txt', 'w') as file:
#     for f in all_files:
#         file.write(str(f) + '\n')


# os.chdir('./data')
# os.system('kaggle competitions download g2net-detecting-continuous-gravitational-waves -f sample_submission.csv')

GODataGenerator().generate_signals(15)
