from fab_docker_gpu.gpu_deploy import *



DOCKER_DIR = abspath('./docker')
SCRIPTS_DIR = abspath('./scripts')
ENV_PATH = abspath('.env')


d = Deploy(DOCKER_DIR, SCRIPTS_DIR, ENV_PATH)

def deploy_train(n=10, gpus=1, token=None):
    d.deploy('train', 'train', n, gpus, token)

def stop_train(script=None):
    d.stop('train', script)
