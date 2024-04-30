import subprocess

def test():
    subprocess.run('python "main.py" --problem test --case finetune --model "testii box64 skip32" --data_raw test_2dp_domain --data_prep testii --notes "try"')
    subprocess.run('python "main.py" --problem test --case test --model "testii box64 skip32" --data_raw test_2dp_domain --data_prep testii --notes "try"')
    subprocess.run('python "main.py" --problem test --case test --model "test_2dp_domain inputs_gksi box64 skip32" --data_raw test_2dp_domain --data_prep testii --notes "try"')
    subprocess.run('python "main.py" --problem test --data_raw test_2dp_domain --notes "try"')

