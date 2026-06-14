## Git submodules

Used here for repos that dont offer pypi packages for model use.
Pinned to specific version of the repo for reproducability.

#### Git stuff

**Info about submodule + commit:**

```bash
git submodule status
```

**When cloning main repo use:**

```bash
git clone --recurse-submodules ...
```

#### To update:

```bash
cd /tab/external/limix
git fetch
git checkout main
git pull
cd /tab
git status
```
