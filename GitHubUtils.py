import os
import git
import shutil
 
def configure_ssh_key(env_path):
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv(env_path)
    ssh_key = os.getenv('SSH_PRIVATE_KEY')
    ssh_key_path = os.path.expanduser('~/.ssh/id_rsa')
    os.makedirs(os.path.dirname(ssh_key_path), exist_ok=True)
    with open(ssh_key_path, 'w') as f:
        f.write(ssh_key)
    os.chmod(ssh_key_path, 0o600)
    os.system('ssh-keyscan github.com >> ~/.ssh/known_hosts')
 
def clone_repo(repo_url, branch):
    repo_dir = os.path.join(os.getcwd(), 'repo')
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    git.Repo.clone_from(repo_url, repo_dir, branch=branch)
    return repo_dir
 
def ensure_archive_dir(repo_dir):
    archive_dir = os.path.join(repo_dir, 'model_archive')
    os.makedirs(archive_dir, exist_ok=True)
    return archive_dir
 
def archive_existing_model(repo_dir):
    model_path = os.path.join(repo_dir, 'model.h5')
    archive_dir = ensure_archive_dir(repo_dir)
    if os.path.exists(model_path):
        archive_path = os.path.join(archive_dir, 'model.h5')
        shutil.move(model_path, archive_path)
 
def update_weights(repo_dir, model_id):
    model_path = os.path.join(os.getcwd(), 'model.h5')
    new_model_path = os.path.join(repo_dir, 'model.h5')
    shutil.copy(model_path, new_model_path)
 
def commit_and_push(repo_dir, branch):
    repo = git.Repo(repo_dir)
    repo.git.add('model.h5')
    repo.index.commit('Update model with latest weights')
    origin = repo.remote(name='origin')
    origin.push(refspec=f'HEAD:{branch}')
 
def cleanup_repo(repo_dir):
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
 
def update_model(model_id, env_path, repo_url, branch, project_name):
    configure_ssh_key(env_path)
    repo_dir = clone_repo(repo_url, branch)
    archive_existing_model(repo_dir)
    update_weights(repo_dir, model_id)
    commit_and_push(repo_dir, branch)
    cleanup_repo(repo_dir)
