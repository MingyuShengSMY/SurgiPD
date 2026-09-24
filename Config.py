import numpy as np


class BaseConfig:
    n_iteration = int(1e4)
    vis_iteration = 100

    learning_rate = 0.001

    initial_epsilon = 1.0
    # epsilon_decay_rate = 0.99
    final_epsilon = 0.1
    epsilon_decay_rate = (initial_epsilon - final_epsilon) / n_iteration * 0.75

    obv_norm_u: np.ndarray
    obv_norm_v: np.ndarray

    batch_size = 1

    memory_buffer_mark = False
    memory_buffer_size = 1
    memory_buffer_clear_mark = False
    memory_buffer_push_num = 1000

    model_update_per_episodes = 1
    model_update_k_epoch = 1
    model_update_k_iter = -1
    model_log_per_iter = 1
    model_start_train_thresh = 1
    imitation_iter = int(5e5)

    model_update_times_approx = n_iteration / model_update_per_episodes * model_update_k_epoch * model_update_per_episodes / model_update_per_episodes
    model_update_iter_approx = n_iteration / model_update_per_episodes

    calculate_gae_mark = False

    gpu = True

    on_policy = True

    fixed_std_for_continuous_action = False

    run_check = False

    load_expert_model = False

    expert_rollout = False

    load_expert_memory_to_rl = False

    load_rollout_memory = False

    GUI_mark = False

    her_mark = False

    obs_normalizer = False

    vision_modal = False

    mp_mark = False
    mp_num = 1

    frame_k = 1

    num_workers = 0





class Q_LearningConfig(BaseConfig):
    n_iteration = int(10000)
    vis_iteration = 100
    # learning_rate = 0.001
    learning_rate = 0.1

    # initial_epsilon = 1.0
    # # epsilon_decay_rate = 0.99
    # final_epsilon = 0.1

    obv_norm_u: np.ndarray
    obv_norm_v: np.ndarray

    memory_buffer_size = 1

    batch_size = 1


class DQNConfig(BaseConfig):
    n_iteration = int(1000000)
    vis_iteration = 10000

    learning_rate = 0.001

    initial_epsilon = 1.0
    # epsilon_decay_rate = 0.99
    final_epsilon = 0.1
    epsilon_decay_rate = (initial_epsilon - final_epsilon) / n_iteration * 0.5

    obv_norm_u: np.ndarray
    obv_norm_v: np.ndarray

    batch_size = 64

    memory_buffer_mark = True
    memory_buffer_size = 10000

    model_update_per_episodes = 1
    model_update_k_epoch = 1
    model_log_per_iter = 10000

    on_policy = False

    gpu = False


class REINFORCEConfig(BaseConfig):
    n_iteration = int(10000)
    vis_iteration = 100

    learning_rate = 0.001

    # initial_epsilon = 1.0
    # epsilon_decay_rate = 0.99
    # final_epsilon = 0.1

    obv_norm_u: np.ndarray
    obv_norm_v: np.ndarray

    batch_size = 64

    memory_buffer_mark = True
    memory_buffer_size = 50000


class PPOConfig(BaseConfig):
    n_iteration = int(1e7)
    vis_iteration = 10000
    # vis_iteration = 1

    learning_rate = 3e-4
    learning_rate2 = 1e-3

    initial_epsilon = 1.0
    # epsilon_decay_rate = 0.99
    final_epsilon = 0.1

    obv_norm_u: np.ndarray
    obv_norm_v: np.ndarray

    batch_size = 64

    memory_buffer_mark = True
    memory_buffer_size = 5000
    memory_buffer_clear_mark = True

    model_update_per_episodes = 2048
    model_update_k_epoch = 10
    model_log_per_iter = 10000

    model_update_times_approx = n_iteration / model_update_per_episodes * model_update_k_epoch * model_update_per_episodes / model_update_per_episodes
    model_update_iter_approx = n_iteration / model_update_per_episodes

    epsilon_decay_rate = (initial_epsilon - final_epsilon) / model_update_iter_approx * 0.5

    calculate_gae_mark = True

    gpu = True

    critic_loss_lambda = 0.5
    entropy_loss_lambda = 0.01

    fixed_std_for_continuous_action = True


class DDPGConfig(BaseConfig):
    on_policy = False

    n_iteration = int(1e6)
    vis_iteration = 10000
    # vis_iteration = 1

    learning_rate = 1e-3
    learning_rate2 = 1e-3

    initial_epsilon = 0.1
    epsilon_decay_rate = 0.99
    final_epsilon = 0.1

    obv_norm_u: np.ndarray
    obv_norm_v: np.ndarray

    batch_size = 256

    memory_buffer_mark = True
    memory_buffer_size = int(5e5)
    memory_buffer_clear_mark = False

    model_update_per_episodes = 1
    model_update_k_epoch = 1
    model_log_per_iter = 10000
    model_start_train_thresh = 5e3
    # model_start_train_thresh = 300
    imitation_iter = int(1e4) * 0
    q_only_iter = int(0)

    expert_rollout = True

    load_expert_model = False

    load_expert_memory_to_rl = False

    # load_rollout_memory = False
    load_rollout_memory = True

    calculate_gae_mark = False

    run_check = False

    gpu = True

    GUI_mark = False
    # GUI_mark = True

    her_mark = True

    # obs_normalizer = False
    obs_normalizer = True


class SACConfig(BaseConfig):
    on_policy = False

    n_iteration = int(1e6)
    vis_iteration = 10000
    # vis_iteration = 1

    learning_rate = 1e-4
    learning_rate2 = 1e-4

    initial_epsilon = 1.0  # alpha
    epsilon_decay_rate = 0.99
    final_epsilon = 0.1

    obv_norm_u: np.ndarray
    obv_norm_v: np.ndarray

    batch_size = 256

    memory_buffer_mark = True
    memory_buffer_size = int(1e6)
    memory_buffer_clear_mark = False
    memory_buffer_push_num = 1

    model_update_per_episodes = 1
    model_update_k_epoch = 1
    model_update_k_iter = 1
    model_log_per_iter = int(n_iteration * 0.01)
    model_start_train_thresh = int(n_iteration * 0.005)
    # model_start_train_thresh = 5e3
    # model_start_train_thresh = 300
    imitation_iter = int(1e4) * 0
    q_only_iter = int(0)

    # expert_rollout = False
    expert_rollout = True

    load_expert_model = False

    load_expert_memory_to_rl = False

    load_rollout_memory = False
    # load_rollout_memory = True

    calculate_gae_mark = False

    run_check = False

    gpu = True

    GUI_mark = False
    # GUI_mark = True

    her_mark = False
    # her_mark = True

    obs_normalizer = False
    # obs_normalizer = True

    mp_mark = True
    mp_num = 1


class SAC_CV_Config(BaseConfig):
    on_policy = False

    n_iteration = int(3e6)
    # n_iteration = int(2e6)
    vis_iteration = 10000
    # vis_iteration = 1

    learning_rate = 1e-4
    learning_rate2 = 1e-4

    initial_epsilon = 1.0  # alpha
    epsilon_decay_rate = 0.99
    final_epsilon = 0.1

    obv_norm_u: np.ndarray
    obv_norm_v: np.ndarray

    batch_size = 256

    memory_buffer_mark = True
    memory_buffer_size = int(1e6)
    memory_buffer_clear_mark = False
    memory_buffer_push_num = 1

    model_update_per_episodes = 1
    model_update_k_epoch = 1
    model_update_k_iter = 1
    model_log_per_iter = min(int(n_iteration * 0.01), 1000)
    # model_log_per_iter = 1
    model_start_train_thresh = min(int(n_iteration * 0.01), 10000)
    # model_start_train_thresh = 5e3
    # model_start_train_thresh = 300
    imitation_iter = int(1e4) * 0
    q_only_iter = int(0)

    # expert_rollout = False
    expert_rollout = True

    load_expert_model = False

    load_expert_memory_to_rl = False

    load_rollout_memory = False
    # load_rollout_memory = True

    calculate_gae_mark = False

    run_check = False
    # run_check = True

    gpu = True

    GUI_mark = False
    # GUI_mark = True

    her_mark = False
    # her_mark = True

    obs_normalizer = False
    # obs_normalizer = True

    # vision_modal = False
    vision_modal = True

    # img_extractor_backbone = "flatten"
    # img_extractor_backbone = "cnn"
    img_extractor_backbone = "resnet18"
    # img_extractor_backbone = "resnet18_conv1"
    # img_extractor_backbone = "vgg"

    mp_num = 1

    frame_k = 3

    num_workers = 12




class DrQConfig(BaseConfig):
    on_policy = False

    n_iteration = int(3e6)
    # n_iteration = int(2e6)
    vis_iteration = 10000
    # vis_iteration = 1

    learning_rate = 1e-4
    learning_rate2 = 1e-4

    initial_epsilon = 1.0  # alpha
    epsilon_decay_rate = 0.99
    final_epsilon = 0.1

    obv_norm_u: np.ndarray
    obv_norm_v: np.ndarray

    batch_size = 256

    memory_buffer_mark = True
    memory_buffer_size = int(1e6)
    memory_buffer_clear_mark = False
    memory_buffer_push_num = 1

    model_update_per_episodes = 1
    model_update_k_epoch = 1
    model_update_k_iter = 1
    model_log_per_iter = min(int(n_iteration * 0.01), 10000)
    # model_log_per_iter = 1
    model_start_train_thresh = min(int(n_iteration * 0.01), 10000)
    # model_start_train_thresh = 5e3
    # model_start_train_thresh = 300
    imitation_iter = int(1e4) * 0
    q_only_iter = int(0)

    # expert_rollout = False
    expert_rollout = True

    load_expert_model = False

    load_expert_memory_to_rl = False

    load_rollout_memory = False
    # load_rollout_memory = True

    calculate_gae_mark = False

    run_check = False
    # run_check = True

    gpu = True

    GUI_mark = False
    # GUI_mark = True

    her_mark = False
    # her_mark = True

    obs_normalizer = False
    # obs_normalizer = True

    # vision_modal = False
    vision_modal = True

    mp_num = 1

    frame_k = 3

    num_workers = 12












