"""
学習済みのパラメータを使って動画を作成
gymのバージョン関係のためFFmpegを利用
"""

import os
import sys
import argparse
from pathlib import Path
import gym
import numpy as np
from stable_baselines3 import PPO  # 使用したアルゴリズムに合わせて変更してください
from PIL import Image  # 画像保存のためにPillowをインポート

# パスの設定（必要に応じて調整してください）
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

# ユーザー設定のパスをインポート
from user_settings import EXPERIMENT_ABS_PATH

# 環境のインポート関数
from somo_rl.utils.import_environment import import_env
from somo_rl.utils import parse_config


def save_rgb_array(rgb_array, filename):
    """RGB配列を画像として保存"""
    img = Image.fromarray(rgb_array)
    img.save(filename)


def record_video(
    exp_abs_path,
    exp_name,
    run_group_name,
    run_name,
    output_video_path,
    video_length=500
):
    run_ID = [exp_name, run_group_name, run_name]

    run_dir = Path(exp_abs_path)
    for subdivision in run_ID:
        run_dir = run_dir / subdivision

    # run_configの読み込み
    run_config_file = run_dir / "run_config.yaml"
    run_config = parse_config.validate_config(run_config_file)
    if not run_config:
        print('ERROR: Run "' + run_name + '" invalid run config')
        return 1

    # 環境のインポート
    env_id = run_config["env_id"]
    import_env(env_id)

    # 学習済みモデルの読み込み
    models_dir = run_dir / "models"
    model_path = models_dir / "best_model.zip"  # best_model.zipを使用

    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        return 1

    # モデルのロード
    model = PPO.load(str(model_path))

    # 環境の作成
    def make_env():
        env = gym.make(
            env_id,
            run_config=run_config,
        )
        env.render_mode = "rgb_array"  # render_modeを明示的に設定
        env._max_episode_steps = run_config["max_episode_steps"]
        return env

    # 環境の作成と動画録画の設定
    env = make_env()

    # 動画保存先ディレクトリを指定された場所に設定
    video_folder = run_dir / "video"
    video_folder.mkdir(parents=True, exist_ok=True)  # 親ディレクトリも作成

    # 動画録画用にMonitorを使用
    from gym.wrappers import Monitor
    env = Monitor(
        env,
        str(video_folder),
        video_callable=None,  # 全てのエピソードを録画
        force=True,
    )

    # モデルの評価と動画の録画
    obs = env.reset()
    for step in range(video_length):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        # フレームを保存するためにrenderメソッドを呼び出す
        rgb_array = env.render(mode="rgb_array")
        save_rgb_array(rgb_array, f"{video_folder}/frame_{step:05d}.png")

        if done:
            obs = env.reset()
            break
    env.close()

    print(f"Frames saved to {video_folder}")
    ffmpeg_command = f"ffmpeg -framerate 30 -i \"{video_folder}/frame_%05d.png\" -c:v libx264 -pix_fmt yuv420p \"{output_video_path}\""
    print("To create a video from frames, use FFmpeg as follows:")
    print(ffmpeg_command)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record video of trained model")
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
        "-o",
        "--output_video_path",
        help="Output video file name (e.g., 'trained_agent.mp4')",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-l",
        "--video_length",
        help="Length of the recorded video in steps",
        type=int,
        default=500,
    )
    args = parser.parse_args()

    # 絶対パスの設定
    exp_abs_path = EXPERIMENT_ABS_PATH

    output_video_path = Path(args.output_video_path)

    record_video(
        exp_abs_path,
        args.exp_name,
        args.run_group_name,
        args.run_name,
        output_video_path,
        args.video_length
    )
