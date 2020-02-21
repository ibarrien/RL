# Mountcar semi gradient Sarsa agent
## How to run
To run, do the following
```bash
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
$ python mountaincar.py
```
Video output is saved in the directory `./video`.
To stop, make a KeyboardInterrupt (CTRL-C).
To modify the behaviour, change the constants defined in `mountaincar.py`.

## Modify the environment
To make the episode continue for longer than 200 steps (the OpenAI gym default), modify the file `env/lib/gym/envs/__init__.py`, change
```python
register(
    id='MountainCar-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)
```
to
```python
register(
    id='MountainCar-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=10000,
    reward_threshold=-110.0,
)
```
(that is, increase `max_episode_steps`.)