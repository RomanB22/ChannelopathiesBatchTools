# Compress output data folders and copy to neurosim
import os

path = '.'    
folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and f.startswith('v52_manualTune')]
for f in folders:
    print('Compressing and copying %s ...' % (f))
    os.system('tar -cvzf %s %s' % (path+'/'+f+'.tar.gz', path+'/'+f))
    os.system('scp %s %s' % (path+'/'+f+'.tar.gz', 'no:///u/salvadord/Models/m1/data/.'))
    os.system('rm -r -f %s' % (path + '/' + f))
    #os.system('rm -r -f %s' % (path+'/'+'.tar.gz'))
