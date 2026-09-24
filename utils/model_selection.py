import numpy as np

from Models.ATGRL import ATGRL
from Reward import Reward


def model_selection(method_name, obs_dim, act_dim, mid_dim, config, img_size=84, img_c_dim=3, bot_dim=0, mask_class_num=0):
    config.vision_model = False

    extra_keys = []

    if method_name == "ATGRL":
        model = ATGRL(
            obs_dim,
            act_dim,
            mid_dim=mid_dim,
            continue_action=config.continuous_act,
            alpha=config.method.args.alpha,
            # gamma=0.95,
            img_size=img_size,
            img_c_dim=img_c_dim,
            frame_k=config.method.frame_k,
            bot_state_dim=bot_dim,

            rollout_method="state",  # adaptive, state, rgb, fixed

            distillation_loss=config.method.args.dist_lf,
            sac_weights_path=config.method.args.sac_weights_path,
        )
        config.vision_model = config.method.vision_input_mark
    else:
        raise ValueError

    model.get_optimizer(config.opts)

    if config.vision_model:
        extra_keys.append("rgb")

    return model, extra_keys


def game_selection(game_name):
    custom_reward_class = Reward

    reward_bias = 0.0
    reward_scalar = 1.0

    game_teacher_mark = False
    max_game_step = 1000

    extra_obs = 0

    bot_state_dim = 0

    mask_class_num = 0

    if game_name == "NeedleReach":
        reward_bias = 0.0
        reward_scalar = 1.0
        max_game_step = 50
        extra_obs = 2 + 3 + 3
        game_teacher_mark = True
        bot_state_dim = 7
    elif game_name == "NeedlePick":
        reward_bias = 0.0
        reward_scalar = 1.0
        max_game_step = 50
        extra_obs = 2 + 3 * 3 + 3 + 3
        game_teacher_mark = True
        bot_state_dim = 7
        mask_class_num = 7
    elif game_name == "NeedleRegrasp":
        reward_bias = 0.0
        reward_scalar = 1.0
        max_game_step = 50
        extra_obs = 4 + 3 * 3 + 3 + 3
        game_teacher_mark = True
        bot_state_dim = 14
    elif game_name == "PegTransfer":
        max_game_step = 100
        extra_obs = 2 + 3 + 3
        game_teacher_mark = True
        bot_state_dim = 7
    elif game_name == "GauzeRetrieve":
        max_game_step = 50
        extra_obs = 2 + 3 * 3 + 3 + 3
        game_teacher_mark = True
        bot_state_dim = 7
    else:
        pass

    return reward_bias, reward_scalar, max_game_step, extra_obs, custom_reward_class, game_teacher_mark, bot_state_dim, mask_class_num

