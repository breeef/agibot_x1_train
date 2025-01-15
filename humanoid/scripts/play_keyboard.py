import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from datetime import datetime

import pygame
from threading import Thread
# 初始化控制变量
x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
keyboard_use = True

if keyboard_use:
    pygame.init()
    # 创建一个小窗口来接收键盘事件
    screen = pygame.display.set_mode((100, 100))
    pygame.display.set_caption("Keyboard Control")
    exit_flag = False

    def handle_keyboard_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd
        
        while not exit_flag:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit_flag = True
                    
            # 获取当前按下的键
            keys = pygame.key.get_pressed()
            
            # 前后移动 (W/S)
            if keys[pygame.K_w]:
                x_vel_cmd = 1.0
                print("W pressed - Forward")
            elif keys[pygame.K_s]:
                x_vel_cmd = -1.0
                print("S pressed - Backward")
            else:
                x_vel_cmd = 0.0
                
            # 左右移动 (A/D)
            if keys[pygame.K_a]:
                y_vel_cmd = 1.0
                print("A pressed - Left")
            elif keys[pygame.K_d]:
                y_vel_cmd = -1.0
                print("D pressed - Right")
            else:
                y_vel_cmd = 0.0
                
            # 旋转控制 (Q/E)
            if keys[pygame.K_q]:
                yaw_vel_cmd = 1.0
                print("Q pressed - Rotate Left")
            elif keys[pygame.K_e]:
                yaw_vel_cmd = -1.0
                print("E pressed - Rotate Right")
            else:
                yaw_vel_cmd = 0.0
            
            # 打印当前的控制值
            # print(f"Commands: x_vel={x_vel_cmd:.2f}, y_vel={y_vel_cmd:.2f}, yaw_vel={yaw_vel_cmd:.2f}")
            
            pygame.time.delay(100)

    keyboard_thread = Thread(target=handle_keyboard_input)
    keyboard_thread.daemon = True  # 设置为守护线程，这样主程序退出时线程会自动结束
    keyboard_thread.start()

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.env.episode_length_s = 1000
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False 
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.continuous_push = False 
    env_cfg.domain_rand.randomize_base_mass = False 
    env_cfg.domain_rand.randomize_com = False 
    env_cfg.domain_rand.randomize_gains = False 
    env_cfg.domain_rand.randomize_torque = False 
    env_cfg.domain_rand.randomize_link_mass = False 
    env_cfg.domain_rand.randomize_motor_offset = False 
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.noise.curriculum = False
    env_cfg.commands.heading_command = False

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    current_time_str = datetime.now().strftime('%H-%M-%S')
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, '0_exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env_cfg.sim.dt * env_cfg.control.decimation)
    robot_index = 0 # which robot is used for logging
    joint_index = 5 # which joint is used for logging
    stop_state_log = 1000 # number of steps before plotting states
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir,exist_ok=True)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir,exist_ok=True)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))
    
    obs = env.get_observations()

    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    try:
        for i in range(10*stop_state_log):
            actions = policy(obs.detach())
            
            if FIX_COMMAND:
                env.commands[:, 0] = 0.5
                env.commands[:, 1] = 0
                env.commands[:, 2] = 0
                env.commands[:, 3] = 0.
            else:
                # 打印当前的控制命令
                if x_vel_cmd or y_vel_cmd or yaw_vel_cmd:
                    print(f"Applying commands: x={x_vel_cmd:.2f}, y={y_vel_cmd:.2f}, yaw={yaw_vel_cmd:.2f}")
                env.commands[:, 0] = x_vel_cmd
                env.commands[:, 1] = y_vel_cmd
                env.commands[:, 2] = yaw_vel_cmd
                env.commands[:, 3] = 0.
            obs, critic_obs, rews, dones, infos = env.step(actions.detach())

            if RENDER:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
                img = np.reshape(img, (1080, 1920, 4))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video.write(img[..., :3])

            if i > stop_state_log*0.2 and i < stop_state_log:
                dict = {
                        'base_height' : env.root_states[robot_index, 2].item(),
                        'foot_z_l' : env.rigid_state[robot_index,4,2].item(),
                        'foot_z_r' : env.rigid_state[robot_index,9,2].item(),
                        'foot_forcez_l' : env.contact_forces[robot_index,4,2].item(),
                        'foot_forcez_r' : env.contact_forces[robot_index,9,2].item(),
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'command_x': x_vel_cmd,
                        'base_vel_y':  env.base_lin_vel[robot_index, 1].item(),
                        'command_y': y_vel_cmd,
                        'base_vel_z':  env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw':  env.base_ang_vel[robot_index, 2].item(),
                        'command_yaw': yaw_vel_cmd,
                        'dof_pos_target': actions[robot_index, 0].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, 0].item(),
                        'dof_vel': env.dof_vel[robot_index, 0].item(),
                        'dof_torque': env.torques[robot_index, 0].item(),
                        'command_sin': obs[0,0].item(),
                        'command_cos': obs[0,1].item(),
                }

                # add dof_pos_target
                for i in range(env_cfg.env.num_actions):
                    dict[f'dof_pos_target[{i}]'] = actions[robot_index, i].item() * env.cfg.control.action_scale,

                # add dof_pos
                for i in range(env_cfg.env.num_actions):
                    dict[f'dof_pos[{i}]'] = env.dof_pos[robot_index, i].item(),

                # add dof_torque
                for i in range(env_cfg.env.num_actions):
                    dict[f'dof_torque[{i}]'] = env.torques[robot_index, i].item(),

                # add dof_vel
                for i in range(env_cfg.env.num_actions):
                    dict[f'dof_vel[{i}]'] = env.dof_vel[robot_index, i].item(),

                logger.log_states(dict=dict)
            
            elif i == stop_state_log:
                logger.plot_states()
            elif i == stop_state_log:
                logger.plot_states()

            # ====================== Log states ======================
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)

    finally:
        if RENDER:
            video.release()
        # 清理键盘控制相关资源
        if keyboard_use:
            exit_flag = True
            pygame.quit()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RENDER = False
    FIX_COMMAND = False
    args = get_args()
    play(args)
