import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from copy import deepcopy

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from pathlib import Path

from stable_baselines3.common.vec_env import SubprocVecEnv# DummyVecEnv
# SubprocVecEnv(並列処理用) を DummyVecEnv に置き換えることで、シングルプロセス環境でのテストを実施
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    BaseCallback
)

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from somo_rl.utils.import_environment import import_env
from somo_rl.utils import parse_config, construct_policy_model

from user_settings import EXPERIMENT_ABS_PATH
import gym
from somo_rl.utils.plot_results import plot_for_algorithm



# 損失ログ用のカスタムコールバック
class LossLoggingCallback(BaseCallback):
    def __init__(self, log_dir, alg_name, verbose=1):
        super(LossLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.alg_name = alg_name
        self.losses = []

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        # 各ロールアウト終了時に損失を取得
        logger = self.model.logger.name_to_value
        if self.alg_name == "PPO":
            policy_loss = logger.get("train/policy_loss")
            value_loss = logger.get("train/value_loss")
            if policy_loss is not None and value_loss is not None:
                self.losses.append([self.num_timesteps, policy_loss, value_loss])
                if self.verbose > 0:
                    print(f"PPO Losses at timestep {self.num_timesteps}: Policy Loss={policy_loss}, Value Loss={value_loss}")
        elif self.alg_name == "SAC":
            qf_loss = logger.get("train/qf_loss")
            policy_loss = logger.get("train/policy_loss")
            alpha_loss = logger.get("train/alpha_loss")
            if qf_loss is not None and policy_loss is not None and alpha_loss is not None:
                self.losses.append([self.num_timesteps, qf_loss, policy_loss, alpha_loss])
                if self.verbose > 0:
                    print(f"SAC Losses at timestep {self.num_timesteps}: QF Loss={qf_loss}, Policy Loss={policy_loss}, Alpha Loss={alpha_loss}")
        elif self.alg_name == "TD3":
            qf1_loss = logger.get("train/qf1_loss")
            qf2_loss = logger.get("train/qf2_loss")
            policy_loss = logger.get("train/policy_loss")
            if qf1_loss is not None and qf2_loss is not None and policy_loss is not None:
                self.losses.append([self.num_timesteps, qf1_loss, qf2_loss, policy_loss])
                if self.verbose > 0:
                    print(f"TD3 Losses at timestep {self.num_timesteps}: QF1 Loss={qf1_loss}, QF2 Loss={qf2_loss}, Policy Loss={policy_loss}")

    def _on_training_end(self) -> None:
        # 学習終了時に損失をCSVファイルに保存
        if self.alg_name == "PPO":
            loss_file = os.path.join(self.log_dir, 'ppo_losses.csv')
            header = 'timesteps,policy_loss,value_loss'
        elif self.alg_name == "SAC":
            loss_file = os.path.join(self.log_dir, 'sac_losses.csv')
            header = 'timesteps,qf_loss,policy_loss,alpha_loss'
        elif self.alg_name == "TD3":
            loss_file = os.path.join(self.log_dir, 'td3_losses.csv')
            header = 'timesteps,qf1_loss,qf2_loss,policy_loss'
        np.savetxt(loss_file, np.array(self.losses), delimiter=',', header=header, comments='')
        if self.verbose > 0:
            print(f"Saved losses to {loss_file}")


# モニターファイルを集約する関数
def aggregate_monitors(monitoring_dir, num_envs):
    data = []
    for i in range(num_envs):
        monitor_path = monitoring_dir / f"{i}.monitor.csv"
        if monitor_path.exists():
            try:
                df = pd.read_csv(monitor_path, skiprows=1)  # 最初の行はコメントとして飛ばす
                df['env_id'] = i  # 環境IDを追加
                data.append(df)
            except Exception as e:
                print(f"Failed to read {monitor_path}: {e}")
    if data:
        combined = pd.concat(data, ignore_index=True)
        return combined
    return None


# 追加のコールバッククラスをここに定義
class AlphaLoggingCallback(BaseCallback):
    def __init__(self, log_dir, verbose=1):
        super(AlphaLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.alphas = []

    def _on_step(self) -> bool:
        if hasattr(self.model, 'alpha'):
            alpha = getattr(self.model, 'alpha')
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            self.alphas.append([self.num_timesteps, alpha])
            if self.verbose > 0:
                print(f"Alpha at timestep {self.num_timesteps}: {alpha}")
        return True

    def _on_training_end(self) -> None:
        alpha_file = os.path.join(self.log_dir, 'alpha.csv')
        header = 'timesteps,alpha'
        np.savetxt(alpha_file, np.array(self.alphas), delimiter=',', header=header, comments='')
        if self.verbose > 0:
            print(f"Saved alpha values to {alpha_file}")

class ActionVarianceLoggingCallback(BaseCallback):
    def __init__(self, log_dir, verbose=1):
        super(ActionVarianceLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.variances = []

    def _on_step(self) -> bool:
        if hasattr(self.model, 'action_variance'):
            action_variance = getattr(self.model, 'action_variance')
            if isinstance(action_variance, torch.Tensor):
                action_variance = action_variance.item()
            self.variances.append([self.num_timesteps, action_variance])
            if self.verbose > 0:
                print(f"Action Variance at timestep {self.num_timesteps}: {action_variance}")
        return True

    def _on_training_end(self) -> None:
        variance_file = os.path.join(self.log_dir, 'action_variance.csv')
        header = 'timesteps,action_variance'
        np.savetxt(variance_file, np.array(self.variances), delimiter=',', header=header, comments='')
        if self.verbose > 0:
            print(f"Saved action variance to {variance_file}")

class TargetPolicyNoiseLoggingCallback(BaseCallback):
    def __init__(self, log_dir, verbose=1):
        super(TargetPolicyNoiseLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.noises = []

    def _on_step(self) -> bool:
        if hasattr(self.model, 'target_policy_noise'):
            target_policy_noise = getattr(self.model, 'target_policy_noise')
            if isinstance(target_policy_noise, torch.Tensor):
                target_policy_noise = target_policy_noise.item()
            self.noises.append([self.num_timesteps, target_policy_noise])
            if self.verbose > 0:
                print(f"Target Policy Noise at timestep {self.num_timesteps}: {target_policy_noise}")
        return True

    def _on_training_end(self) -> None:
        noise_file = os.path.join(self.log_dir, 'target_policy_noise.csv')
        header = 'timesteps,target_policy_noise'
        np.savetxt(noise_file, np.array(self.noises), delimiter=',', header=header, comments='')
        if self.verbose > 0:
            print(f"Saved target policy noise to {noise_file}")




def create_note(run_dir, run_ID, start_datetime, note):
    try:
        git_commit_hash = str(
            subprocess.check_output(["git", "rev-parse", "HEAD"])
        ).strip()[2:-3]
    except:
        git_commit_hash = "unavailable"

    with open(run_dir / "info.txt", "w") as info_file:
        info_file.write("_____________________________________\n")
        info_file.write("TRAINING RUN INFO:\n")
        info_file.write(f"- Run ID: {run_ID}\n")
        info_file.write("- PID: " + str(os.getpid()) + "\n")
        info_file.write("- Start Datetime: " + start_datetime + "\n")
        info_file.write("- Git Commit Hash: " + git_commit_hash + "\n")

        info_file.write("_____________________________________\n")
        info_file.write("NOTES ON EXPERIMENT:\n")
        info_file.write("- " + note + "\n")


def make_env(
    env_id,
    run_config,
    max_episode_steps,
    rank=0,
    run_ID=None,
    monitoring_dir=None,
    render=False,
    debug=False,
    is_eval_env=False,
):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param monitoring_dir: (str) directory to store monitor files
    """

    def _init():
        try:
            seed = run_config["seed"] + rank
        except:
            seed = rank
        if is_eval_env:  # set outside reasonable range of # env ranks
            seed += 100
        set_random_seed(seed)
        print(f"Set seed to {seed}.")

        import_env(run_config["env_id"])
        env = gym.make(
            env_id,
            run_config=run_config,
            run_ID=run_ID,
            render=render,
            debug=debug,
        )
        env._max_episode_steps = max_episode_steps
        env.seed(seed)
        if monitoring_dir is not None and not is_eval_env:
            log_file = Path(monitoring_dir) / str(rank)
            reward_keywords = tuple([])
            if run_config["reward_flags"]:
                reward_keywords = tuple(run_config["reward_flags"].keys())
            env = Monitor(env, str(log_file), info_keywords=reward_keywords)
        return env

    return _init


def log_on_complete(
    start_time, run_ID, eval_dir, run_dir, run_config, success=True
):
    # Log training run results
    end_time = time.time()
    end_datetime = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    training_duration = end_time - start_time

    print(f"Training Duration: {training_duration}")

    try:
        best_eval_reward = np.max(
            np.mean(
                dict(np.load(Path(eval_dir) / "evaluations.npz"))["results"], axis=1
            )
        )
    except:
        best_eval_reward = None

    with open(run_dir / "info.txt", "a") as info_file:
        eval_reward_message = (
            "No eval callbacks generated."
            if best_eval_reward is None
            else best_eval_reward
        )
        info_file.write("_____________________________________\n")
        info_file.write("RESULTS:\n")
        info_file.write("- Success: " + str(success) + "\n")
        info_file.write("- Best Eval Reward: " + str(eval_reward_message) + "\n")
        info_file.write("- End Datetime: " + end_datetime + "\n")
        info_file.write(
            "- Training Duration (sec): " + str(int(np.round(training_duration))) + "\n"
        )
        info_file.write(
            "- Training Timesteps: " + str(run_config["training_timesteps"]) + "\n"
        )

    if success:
        print(
            f"\nSUCCESS! Experiment {run_ID} is done training!\n"
        )
    else:
        print(
            f"\nERROR! Experiment {run_ID} failed on following exception:\n"
        )



def run(
    exp_abs_path,
    exp_name,
    run_group_name,
    run_name,
    render=False,
    debug=False,
    overwrite=False,
    note="",
    run_config_input=None,
    expert_dir_abs_path=None,
):

    run_ID = [exp_name, run_group_name, run_name]

    run_dir = Path(exp_abs_path)
    for subdivision in run_ID:
        run_dir = run_dir / subdivision

    # run_config_input is a run_config dictionary
    if isinstance(run_config_input, dict):
        run_config = deepcopy(run_config_input)
    # run_config_input wasn't given or is a path
    else:
        # use the run_config found in the run directory
        if run_config_input is None:
            run_config_file = run_dir / "run_config.yaml"
        # run_config_input is a path to the run_config.yaml file
        else:
            run_config_file = Path(run_config_input)

        run_config = parse_config.validate_config(run_config_file)

        if not overwrite:
            if "run_started" in run_config.keys() and run_config["run_started"]:
                print(f"CRITICAL WARNING: Run {run_ID} already started/complete. Edit 'run_started' config field to allow overwrite.")
                return 1

            with open(run_config_file, "a") as config_file:
                config_file.write("\nrun_started: True")

    if not run_config:
        print('ERROR: Run "' + run_name + '" invalid run config')
        return 1

    import_env(run_config["env_id"])

    if expert_dir_abs_path:
        run_config["expert_dir_abs_path"] = expert_dir_abs_path

    # Set up logging and results directories
    monitoring_dir = run_dir / "monitoring"
    models_dir = run_dir / "models"
    checkpoints_dir = run_dir / "callbacks" / "checkpoints"
    eval_dir = run_dir / "callbacks" / "eval_results"
    results_dir = run_dir / "results"

    dirs = [monitoring_dir, models_dir, checkpoints_dir, eval_dir, results_dir]

    for dir_path in dirs:
        shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path)

    tensorboard_log = None
    if "tensorboard_log" in run_config.keys():
        tensorboard_log = run_config["tensorboard_log"]

    env_id = run_config["env_id"]

    # Number of processes to use
    if "num_threads" in run_config:
        num_threads = run_config["num_threads"]
    else:
        num_threads = 1

    start_time = time.time()
    start_datetime = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

    create_note(run_dir, run_ID, start_datetime, note)

    # Create the vectorized environment
    train_env =  SubprocVecEnv(
        [
            make_env(
                env_id=env_id,
                run_config=run_config,
                max_episode_steps=run_config["max_episode_steps"],
                rank=i,
                #run_ID=run_ID,
                run_ID=run_ID + [f"env_{i}"],  # ユニークなrun_IDを追加
                monitoring_dir=monitoring_dir,
                render=(render and i == 0),
                debug=debug if i == 0 else False,
            )
            for i in range(num_threads)
        ],
        start_method= "spawn", #"forkserver", forkserverはLinuxやMac環境でのみ使用可
        # start_methodsの設定はSubprocVecEnv関数のみで使用可、DummyVecEnvは不要 
    )

    # separate evaluation env
    eval_env = SubprocVecEnv(
        [
            make_env(
                env_id=env_id,
                run_config=run_config,
                max_episode_steps=run_config["max_episode_steps"],
                #run_ID = deepcopy(run_ID).append("EVAL_ENV")
                rank=0,  # 評価環境は単一プロセス
                run_ID = run_ID + ["EVAL_ENV"],
                monitoring_dir=eval_dir,
                render=False,
                debug=False,
                is_eval_env=True,
            )
        ]
    )

    # create callbacks
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=run_config["eval_cb"]["n_eval_episodes"],
        best_model_save_path=models_dir,
        log_path=eval_dir,
        eval_freq=run_config["eval_cb"]["eval_freq"],
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=run_config["checkpoint_cb"]["save_freq"],
        save_path=checkpoints_dir,
    )
    # 損失をログするコールバックを追加
    loss_logging_callback = LossLoggingCallback(results_dir, run_config["alg"])# resultsディレクトリを参照する
    # 追加のメトリクス用コールバックを追加
    additional_callbacks = []
    if run_config["alg"] == "SAC":
        alpha_logging_callback = AlphaLoggingCallback(monitoring_dir)
        additional_callbacks.append(alpha_logging_callback)
    elif run_config["alg"] == "TD3":
        action_variance_logging_callback = ActionVarianceLoggingCallback(monitoring_dir)
        target_policy_noise_logging_callback = TargetPolicyNoiseLoggingCallback(monitoring_dir)
        additional_callbacks.extend([action_variance_logging_callback, target_policy_noise_logging_callback])

    # コールバックリストに追加コールバックを含める
    callback_list = [eval_callback, checkpoint_callback, loss_logging_callback] + additional_callbacks
    callback = CallbackList(callback_list)

    policy_kwargs = {}
    if "policy_kwargs" in run_config:
        policy_kwargs = deepcopy(run_config["policy_kwargs"])

    model = construct_policy_model.construct_policy_model(
        run_config["alg"],
        run_config["policy"],
        train_env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1
    )

    # Run Training
    try:
        model.learn(total_timesteps=run_config["training_timesteps"], callback=callback)
    except Exception as e:
        log_on_complete(
            start_time, run_ID, eval_dir, run_dir, run_config, success=False
        )
        raise e

    log_on_complete(start_time, run_ID, eval_dir, run_dir, run_config)

    model.save(models_dir / "final_model")
    train_env.close()
    eval_env.close()# 
    
    # トレーニング後にモニターファイルを集約して可視化および保存を行う
    monitoring_data = aggregate_monitors(monitoring_dir, num_threads)
    if monitoring_data is not None:
        plot_for_algorithm(results_dir, monitoring_data, run_config["alg"])
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training run argument parser")
    parser.add_argument(
        "-e",
        "--exp_name",
        help="Experiment name",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-g",
        "--run_group_name",
        help="Run-group name",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-r",
        "--run_name",
        help="Run Name",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-v",
        "--render",
        help="Render the env of one of the threads",
        action="store_true",
    )
    parser.add_argument(
        "-d", "--debug", help="Display SoMo-RL Debugging Dashboard", action="store_true"
    )
    parser.add_argument('-dl','--debug_list', nargs='+', help='List of debugger components to show in panel (space separated). Choose from reward_components, observations, actions, applied_torques', required=False, default=[])
    parser.add_argument(
        "-o",
        "--overwrite",
        help="Allow overwrite of data of previous experiment",
        action="store_true",
    )
    parser.add_argument(
        "--note", help="Note to keep track of purpose of run", default=""
    )
    argument = parser.parse_args()

    debug = argument.debug 
    if len(argument.debug_list) > 0:
        debug = deepcopy(argument.debug_list)

    run(
        EXPERIMENT_ABS_PATH,
        argument.exp_name,
        argument.run_group_name,
        argument.run_name,
        argument.render,
        debug,
        argument.overwrite,
        argument.note,
    )
