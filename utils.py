# Functions for testing trained agents and visualizing test performance.

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env


def make_video(env, agent):
    """
    Captures video of agent performance for a single episode.
    """
    env = wrap_env(env)
    rewards = 0
    steps = 0
    done = False
    state = env.reset()
    while not done:
        env.render()
        action = agent.get_action(state,0)
        state, reward, done, info= env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))


def show_video():
  """
  Displays video of agent test.
  """
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else:
    print("Could not find video")
