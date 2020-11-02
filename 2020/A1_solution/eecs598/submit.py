import os
import zipfile


_A1_FILES = [
  'pytorch101.py',
  'pytorch101.ipynb',
  'knn.py',
  'knn.ipynb',
]


def make_a1_submission(assignment_path):
  _make_submission(assignment_path, _A1_FILES, 'A1')


def _make_submission(assignment_path, file_list, assignment_no):
  uniquename, umid = _get_user_info()
  zip_path = '%s_%s_{}.zip' % (uniquename, umid, assignment_no)
  zip_path = os.path.join(assignment_path, zip_path)
  print('Writing zip file to: ', zip_path)
  with zipfile.ZipFile(zip_path, 'w') as zf:
    for filename in file_list:
      in_path = os.path.join(assignment_path, filename)
      if not os.path.isfile(in_path):
        raise ValueError('Could not find file "%s"' % filename)
      zf.write(in_path, filename)


def _get_user_info():
  uniquename = input('Enter your uniquename (e.g. justincj): ')
  umid = input('Enter your umid (e.g. 12345678): ')
  return uniquename, umid
