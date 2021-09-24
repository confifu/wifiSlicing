import click
from ns3gym import ns3env
from tqdm import tqdm
from Model import ModelHelper
import torch

@click.command()
@click.pass_context
@click.option('--start_sim', type=bool, default=True, help='Start simulation', show_default=True)
@click.option('--iterations', type=int, default=10, help='Number of iterations', show_default=True)
@click.option('--sim_time', type=int, default = 20, help='Simulation time in seconds', show_default=True)
@click.option('--resume_from', help='Checkpoint from which to resume', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the new checkpoint', type=str, required=False, metavar='DIR')
@click.option('--debug', type=bool, default=False, help='Print debug outputs', show_default=True)
def runSimulation(
    ctx: click.Context,
    start_sim: bool,
    iterations: int,
    sim_time: int,
    resume_from: str,
    outdir: str,
    debug: bool
):
    """
    Train the model.
    """

    port = 5555
    seed = 0
    step_time = 1.0       # Do not change
    simArgs = {"--simulationTime": sim_time,
            "--testArg": 123}
    
    env = ns3env.Ns3Env(port=port, stepTime=step_time, startSim=start_sim, simSeed=seed, simArgs=simArgs, debug=debug)
    # simpler:
    #env = ns3env.Ns3Env()
    env.reset()

    if debug:
        ob_space = env.observation_space
        ac_space = env.action_space
        print("Obseration space", ob_space)
        print("Action space", ac_space)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    helper = ModelHelper(device)
    try:
        for currIt in tqdm(range(iterations)):
            obs = env.reset()
            stepIdx = 0
            done = False
            pbar = tqdm(total=sim_time // step_time)
            while not done:
                #Update calls start from 2 secs.
                if stepIdx < 2:
                    action = env.action_space.sample()
                    obs_cur, _, done, _ = env.step(action)

                else:
                    actionTuple = helper.getActionTuple(obs_prev, action)
                    action = helper.getActionFromActionTuple(actionTuple, action)
                    obs_cur, _, done, _ = env.step(action)

                    #sas' (reward is a funciton of s in this case)
                    helper.trainModel(obs_prev, action, actionTuple, obs_cur)
                obs_prev = obs_cur
                stepIdx += 1
                pbar.update(1)
            pbar.close()

    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")
    finally:
        env.close()
        print("Done")


if __name__ == "__main__":
    runSimulation()
