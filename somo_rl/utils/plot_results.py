import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 累積報酬のプロット関数
def plot_cumulative_reward_per_env_combined(results_dir, monitor_data):
    if 'r' not in monitor_data.columns or 'l' not in monitor_data.columns or 'env_id' not in monitor_data.columns:
        print("Monitor data does not contain required columns 'l', 'r', or 'env_id'.")
        return
    
    env_groups = monitor_data.groupby('env_id')
    
    plt.figure()
    for env_id, group in env_groups:
        rewards = group['r'].values
        episodes = np.arange(len(rewards))
        cumulative_rewards = np.cumsum(rewards)
        plt.plot(episodes, cumulative_rewards, label=f'Env {env_id}')
    
    plt.title('Cumulative Rewards per Environment')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cumulative_rewards_per_env.tif'), format='tif')
    plt.savefig(os.path.join(results_dir, 'cumulative_rewards_per_env.jpeg'), format='jpeg')
    plt.close()
    print("Cumulative rewards per environment plotted and saved.")


def plot_average_cumulative_reward(results_dir, monitor_data):
    """
    全環境の平均累積報酬をプロットし、指定されたディレクトリに保存
    """
    if 'r' not in monitor_data.columns or 'l' not in monitor_data.columns:
        print("Monitor data does not contain required columns 'l' and 'r'.")
        return
    
    # エピソード番号を作成
    monitor_data = monitor_data.copy()
    monitor_data['episode'] = monitor_data.groupby('env_id').cumcount()
    
    # エピソードごとに平均累積報酬を計算
    grouped = monitor_data.groupby('episode')['r'].mean()
    episodes = grouped.index.values
    avg_rewards = grouped.values
    avg_cumulative_rewards = np.cumsum(avg_rewards)
    
    plt.figure()
    plt.plot(episodes, avg_cumulative_rewards)
    plt.title('Average Cumulative Rewards Across Environments')
    plt.xlabel('Episodes')
    plt.ylabel('Average Cumulative Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'average_cumulative_rewards.tif'), format='tif')
    plt.savefig(os.path.join(results_dir, 'average_cumulative_rewards.jpeg'), format='jpeg')
    plt.close()
    print("Average cumulative rewards plotted and saved.")



# 損失のプロット関数
def plot_loss(results_dir, loss_file, loss_columns):
    """
    ポリシー損失やバリュー関数損失、Q関数損失の推移をプロットし、.tifおよび.jpeg形式で保存
    """
    if not os.path.exists(loss_file):
        print(f"Loss file {loss_file} does not exist.")
        return
    
    try:
        data = pd.read_csv(loss_file)
    except Exception as e:
        print(f"Failed to read {loss_file}: {e}")
        return

    if 'timesteps' not in data.columns:
        print(f"'timesteps' column not found in {loss_file}.")
        return

    timesteps = data['timesteps']
    
    plt.figure()
    for loss_col in loss_columns:
        if loss_col in data.columns:
            plt.plot(timesteps.values, data[loss_col].values, label=loss_col)
        else:
            print(f"Column {loss_col} not found in {loss_file}.")
    
    plt.title('Losses over Time')
    plt.xlabel('Timesteps')
    plt.ylabel('Loss')
    plt.tight_layout()  # プロット内のレイアウトを自動調整し、枠内に収まるようにする
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'losses.tif'), format='tif')
    plt.savefig(os.path.join(results_dir, 'losses.jpeg'), format='jpeg')
    plt.close()
    print(f"Losses from {loss_file} plotted and saved.")

# エントロピー係数アルファのプロット関数 (SAC向け)
def plot_alpha(results_dir):
    """
    エントロピー係数アルファの推移をプロットし、.tifおよび.jpeg形式で保存
    """
    alpha_file = os.path.join(results_dir, 'alpha.csv')
    if not os.path.exists(alpha_file):
        print(f"Alpha file {alpha_file} does not exist.")
        return

    try:
        data = pd.read_csv(alpha_file)
    except Exception as e:
        print(f"Failed to read {alpha_file}: {e}")
        return

    if 'timesteps' not in data.columns or 'alpha' not in data.columns:
        print(f"'timesteps' or 'alpha' column not found in {alpha_file}.")
        return

    timesteps = data['timesteps']
    alpha_values = data['alpha']

    plt.figure()
    plt.plot(timesteps.values, alpha_values.values)
    plt.title('Alpha over Time')
    plt.xlabel('Timesteps')
    plt.ylabel('Alpha')
    plt.tight_layout()  # プロット内のレイアウトを自動調整し、枠内に収まるようにする
    plt.savefig(os.path.join(results_dir, 'alpha.tif'), format='tif')
    plt.savefig(os.path.join(results_dir, 'alpha.jpeg'), format='jpeg')
    plt.close()
    print("Alpha plotted and saved.")

# アクション分散のプロット関数 (TD3向け)
def plot_action_variance(results_dir):
    """
    アクションの分散の推移をプロットし、.tifおよび.jpeg形式で保存
    """
    action_variance_file = os.path.join(results_dir, 'action_variance.csv')
    if not os.path.exists(action_variance_file):
        print(f"Action variance file {action_variance_file} does not exist.")
        return

    try:
        data = pd.read_csv(action_variance_file)
    except Exception as e:
        print(f"Failed to read {action_variance_file}: {e}")
        return

    if 'timesteps' not in data.columns or 'action_variance' not in data.columns:
        print(f"'timesteps' or 'action_variance' column not found in {action_variance_file}.")
        return

    timesteps = data['timesteps']
    action_variance_values = data['action_variance']

    plt.figure()
    plt.plot(timesteps, action_variance_values.values)
    plt.title('Action Variance over Time')
    plt.xlabel('Timesteps')
    plt.ylabel('Action Variance')
    plt.tight_layout()  # プロット内のレイアウトを自動調整し、枠内に収まるようにする
    plt.savefig(os.path.join(results_dir, 'action_variance.tif'), format='tif')
    plt.savefig(os.path.join(results_dir, 'action_variance.jpeg'), format='jpeg')
    plt.close()
    print("Action variance plotted and saved.")

# ターゲットポリシーノイズのプロット関数 (TD3向け)
def plot_target_policy_noise(results_dir):
    """
    ターゲットポリシーノイズの推移をプロットし、.tifおよび.jpeg形式で保存
    """
    target_policy_noise_file = os.path.join(results_dir, 'target_policy_noise.csv')
    if not os.path.exists(target_policy_noise_file):
        print(f"Target policy noise file {target_policy_noise_file} does not exist.")
        return

    try:
        data = pd.read_csv(target_policy_noise_file)
    except Exception as e:
        print(f"Failed to read {target_policy_noise_file}: {e}")
        return

    if 'timesteps' not in data.columns or 'target_policy_noise' not in data.columns:
        print(f"'timesteps' or 'target_policy_noise' column not found in {target_policy_noise_file}.")
        return

    timesteps = data['timesteps']
    target_policy_noise_values = data['target_policy_noise']

    plt.figure()
    plt.plot(timesteps, target_policy_noise_values.values)
    plt.title('Target Policy Noise over Time')
    plt.xlabel('Timesteps')
    plt.ylabel('Target Policy Noise')
    plt.tight_layout()  # プロット内のレイアウトを自動調整し、枠内に収まるようにする
    plt.savefig(os.path.join(results_dir, 'target_policy_noise.tif'), format='tif')
    plt.savefig(os.path.join(results_dir, 'target_policy_noise.jpeg'), format='jpeg')
    plt.close()
    print("Target policy noise plotted and saved.")

# 使用するアルゴリズムに応じたプロット関数を決定
def plot_for_algorithm(results_dir, monitor_data, alg_name):
    """
    アルゴリズム名に応じて、適切なプロット関数を実行
    """
    # 環境ごとの累積報酬をプロット
    plot_cumulative_reward_per_env_combined(results_dir, monitor_data)
    
    # 平均累積報酬をプロット
    plot_average_cumulative_reward(results_dir, monitor_data)
    
    if alg_name == "PPO":
        ppo_loss_file = os.path.join(results_dir, 'ppo_losses.csv')
        plot_loss(results_dir, ppo_loss_file, ['policy_loss', 'value_loss'])
    elif alg_name == "SAC":
        sac_loss_file = os.path.join(results_dir, 'sac_losses.csv')
        plot_loss(results_dir, sac_loss_file, ['qf_loss', 'policy_loss'])
        plot_alpha(results_dir)
    elif alg_name == "TD3":
        td3_loss_file = os.path.join(results_dir, 'td3_losses.csv')
        plot_loss(results_dir, td3_loss_file, ['qf1_loss', 'qf2_loss', 'policy_loss'])
        plot_action_variance(results_dir)
        plot_target_policy_noise(results_dir)
    else:
        print(f"Algorithm {alg_name} is not supported for plotting.")

