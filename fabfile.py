import fnmatch
from distutils.util import strtobool

from fabric.api import local, abort, run, sudo
from fabric.context_managers import cd, settings, hide, shell_env
from fabric.contrib.console import confirm
import re
from os.path import join
from getpass import getpass

from fabric.operations import put
from fabric.utils import puts

device_re = '.*/dev/nvidia(?P<device>[0-9]+).*'

from fabric.state import env


def with_sudo():
    """
    Prompts and sets the sudo password for all following commands. 
    
    Use like
    
    fab with_sudo command
    """
    env.sudo_password = getpass('Please enter sudo password: ')


def on(which, *args):
    env.hosts = list(filter(lambda x: x not in args, getattr(env, which).split(',')))


def excluding(*args):
    with hide('output', 'running'):
      env.hosts = list(filter(lambda x: x not in args, env.hosts))


def all_machines():
    """
    Sets the machines on which the following commands are executed to the `machines` in `.~/fabricrc`. 
    This overrides the `-H` argument. 
    
    Use like
    
    fab all_machines -P command
    """
    env.hosts = env.machines.split(',')


def get_branch(gitdir):
    """
    Gets the branch of a git directory. 
    
    Args:
        gitdir: path of the git directory 

    Returns: current active branch
        
    """
    with cd(gitdir):
        branch = local('git symbolic-ref --short HEAD', capture=True)
    return branch


def commit_and_push(gitdir, remote='origin', message=None):
    """
    Commits and pushes code changes. 
    
    Args:
        gitdir: git directory 
        remote: remote name (default: origin)
    """
    with cd(gitdir), settings(warn_only=True):
        branch = get_branch(gitdir)
        local('git commit -m "{}" -a'.format('automatic commit' if message is None else message))
        local('git push {} {}'.format(remote, branch))


def clone_code():
    run('mkdir -p {}'.format(env.basedir))
    with cd(env.basedir), settings(warn_only=True):
        run('git clone ' + env.attorch_git)
        if hasattr(env, 'v1_git'):
            run('git clone {} v1_likelihood'.format(env.v1_git))


def pull_code(gitdir=None, branch=None):
    if gitdir is None:
        gitdir = join(env.basedir, '/v1_likelihood')
    with cd(gitdir):
        if branch is None:
            branch = get_branch(gitdir)
        run('git reset --hard')
        run('git clean -fd')
        run('git pull')
        run('git checkout {}'.format(branch))
        run('git pull origin ' + branch)


def sync_env_file():
    local('scp .env ' + env.host_string + ':' + join(env.basedir, 'v1_likelihood/'))


def gpu_devices():
    ret = []
    for dev in run('ls /dev/nvidia[0-9]').split():
        m = re.match(device_re, dev)
        if m:
            ret.append(m.groupdict()['device'])
    return ret


def free_gpu_slots():
    with hide('output', 'running'):
        containers = run('docker ps -q --no-trunc').split()
        gpus = gpu_devices()
        for container in containers:
            host_config = run("docker inspect -f '{{ .HostConfig.Devices }}' " + container)
            match = re.match(device_re, host_config)
            if match:
                dev = match.groupdict()['device']
                if dev in gpus:
                    gpus.remove(dev)
    return gpus


def initialize():
    clone_code()
    sync_env_file()


def availability():
    puts(env.host_string + ' free GPUs: ' + ','.join(free_gpu_slots()))


def build_attorch(pull=True, no_cache=True, push=True):
    build_path = join(env.basedir, 'attorch')
    pull_code(build_path)

    pull = strtobool(str(pull))
    no_cache = strtobool(str(no_cache))
    push = strtobool(str(push))

    image = env.attorch_image

    with cd(build_path):
        args = ''
        if pull:
            args += '--pull '
        if no_cache:
            args += '--no-cache '
        run('docker build {} -f Dockerfile -t {} .'.format(args, image))
    if push:
        with hide('output'):
            run('docker push {} '.format(image))


def deploy(yml_file, service, n=None, rebuild=False):
    project_dir = join(env.basedir, 'v1_likelihood')
    gpus = free_gpu_slots()
    rebuild = strtobool(str(rebuild))

    if len(gpus) == 0:
        puts('No free GPUs found')
        return

    if n is not None:
        gpus = gpus[:n]

    pull_code(project_dir)

    run('docker pull {}'.format(env.attorch_image))
    if 'netgard_image' in env:
        run('docker pull {}'.format(env.netgard_image))

    with cd(project_dir), shell_env(HOSTNAME=env.host_string):
        for gpu in gpus:
            run('nvidia-docker-compose -t {} build {} {}{}'.format(yml_file,
                                                                   '' if not rebuild else '--no-cache',
                                                                   service, gpu))
            run('nvidia-docker-compose -t {} up -d {}{};'.format(yml_file, service, gpu))
    puts('started service {} on {} on GPUs {}'.format(env.host_string, service, ' '.join(gpus)))


def docker_login():
    run('docker login')


def remove_old_containers():
    with settings(warn_only=True):
        run('docker ps -aq| xargs docker rm')


def sync():
    gitdir = env.basedir + '/v1_likelihood/'

    options = " -avz --include '*/' --include '*.py' --exclude '*'"
    for host in env.hosts:
        with settings(warn_only=True):
            local(
                "rsync {options} ./ {host}:{gitdir}".format(
                    host=host,
                    options=options,
                    gitdir=gitdir
                ))

def containers():
    with hide('running'):
        ret = run("docker ps --format '{{.Names}} -> {{.ID}}'")
    return [r.strip() for r in ret.split('\n') if '->' in r]

ps = containers


def stop(wildcard):
    candidates = {k:v for k,v in map(lambda s: s.split(' -> '), containers())}
    selection = fnmatch.filter(candidates.keys(), wildcard)
    stop = [candidates[s] for s in selection]
    if len(stop) > 0:
        run('docker stop {}'.format(' '.join(stop)))


def kill(wildcard):
    candidates = {k:v for k,v in map(lambda s: s.split(' -> '), containers())}
    selection = fnmatch.filter(candidates.keys(), wildcard)
    stop = [candidates[s] for s in selection]
    if len(stop) > 0:
        run('docker kill {}'.format(' '.join(stop)))


def logs(pattern, wildcard='*'):
    candidates = {k:v for k,v in map(lambda s: s.split(' -> '), containers())}
    selection = fnmatch.filter(candidates.keys(), wildcard)
    candidates = [candidates[s] for s in selection]
    for candidate in candidates:
        run('docker logs {} | egrep "{}"'.format(candidate, pattern))


def clean():
    run('docker ps -aq| xargs docker rm')

