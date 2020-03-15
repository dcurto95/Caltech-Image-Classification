import os
import sys

test_name = sys.argv[1]
print("Test:", test_name)
f = open("main_launcher.sh", "w")
f.write("#!/bin/bash\n")

f.write('#SBATCH --job-name="' + test_name + '"\n')

f.write('#SBATCH --qos=debug\n')

f.write('#SBATCH --workdir=.\n')

f.write('#SBATCH --output=../logs/' + test_name + '/model.out\n')

f.write('#SBATCH --error=../logs/' + test_name + '/model.err\n')

f.write('#SBATCH --cpus-per-task=40\n')

f.write('#SBATCH --gres gpu:1\n')

f.write('#SBATCH --time=01:30:00\n')

f.write(
    'module purge; module load ffmpeg/4.0.2 gcc/6.4.0 cuda/9.1 cudnn/7.1.3 openmpi/3.0.0 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.7 szip/2.1.1 opencv/3.4.1 python/3.6.5_ML\n')

f.write('python main.py ' + test_name + '\n')
f.close()

command = os.popen('mkdir ../logs/' + test_name)
print(command.read())
command.close()
print("Folder created.")
command = os.popen('cp cnn.py ../logs/' + test_name)
print(command.read())
command.close()
print("Structure copied.")
command = os.popen('sbatch main_launcher.sh')
print(command.read())
command.close()
