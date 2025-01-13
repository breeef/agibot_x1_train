# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class KQLDHStandCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 66      #all histroy obs num
        short_frame_stack = 5   #short history step
        c_frame_stack = 3  #all histroy privileged obs num
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 73
        single_linvel_index = 53
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24 #episode length in seconds
        use_ref_actions = False
        num_commands = 5 # sin_pos cos_pos vx vy vz

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85


    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kql/urdf/KQL02.urdf'
        # xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x1/mjcf/xyber_x1_flat.xml'

        name = "kql"
        foot_name = "f_row" # max_contact_force  关联，uref中的f_row collison关联 log: PID: 26996 - 20250112_2129 完结散花
        knee_name = "l_lower"
        # 碰撞检测相关节点
        terminate_after_contacts_on = ['Base']
        penalize_contacts_on = ["Base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 5  # starting curriculum state
        platform = 3.
        terrain_dict = {"flat": 0.3, 
                        "rough flat": 0.2,
                        "slope up": 0.2,
                        "slope down": 0.2, 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0, 
                        "stairs up": 0., 
                        "stairs down": 0.,
                        "discrete": 0.1, 
                        "wave": 0.0,}
        terrain_proportions = list(terrain_dict.values())

        rough_flat_range = [0.005, 0.01]  # meter
        slope_range = [0, 0.1]   # rad
        rough_slope_range = [0.005, 0.02]
        stair_width_range = [0.25, 0.25]
        stair_height_range = [0.01, 0.1]
        discrete_height_range = [0.0, 0.01]
        restitution = 0.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.5    # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.02
            dof_vel = 1.5 
            ang_vel = 0.2   
            lin_vel = 0.1   
            quat = 0.1
            gravity = 0.05
            height_measurements = 0.1


    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.]
        #TODO: 换成自己的 此处关联final_swing_joint_delta_pos log: Process ID: 32414 -20250113_1354
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'Ll_root_joint': 0.0, #1 腿根转动 +
            'Ll_upper_joint': 0.0, #2 腿根左右摆动-
            'Ll_knee_joint': -0.11, #3 大腿前后摆动-
            'Ll_lower_joint': 0.23, #4 膝盖前后摆动 +
            'Lf_pitch_joint': -0.11, #5 脚部前后摆动 -
            'Lf_row_joint': 0.0, #6 脚部左右摆动 脚掌 +
            
            'Rl_root_joint': 0.0, #-
            'Rl_upper_joint': 0.0, #+
           'Rl_knee_joint': 0.11, #+
             'Rl_lower_joint': -0.23, #-
            'Rf_pitch_joint': -0.11, #-
            'Rf_row_joint': 0.0, #+
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        stiffness = {'knee_joint': 30, 'upper_joint': 35,'root_joint': 40, 
                     'lower_joint': 100,  'pitch_joint': 35,'row_joint': 35}# 刚度，不明白膝盖的刚度为啥高 log: id 23010-20250112_1643
        damping = {'knee_joint': 3, 'upper_joint': 4.0,'root_joint': 3.0, 
                   'lower_joint':10, 'pitch_joint': 0.5 , 'row_joint': 0.5} ## 增加阻尼

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 50hz 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 200 Hz 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z
     
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.3]
        restitution_range = [0.0, 0.4]

        # push
        push_robots = True
        push_interval_s = 4 # every this second, push robot
        update_step = 2000 * 24 # after this count, increase push_duration index
        push_duration = [0, 0.05, 0.1, 0.15, 0.2, 0.25] # increase push duration during training
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.2

        randomize_base_mass = True
        added_mass_range = [-3, 3] # base mass rand range, base mass is all fix link sum mass

        randomize_com = True
        com_displacement_range = [[-0.05, 0.05],
                                  [-0.05, 0.05],
                                  [-0.05, 0.05]]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  # Factor
        damping_multiplier_range = [0.8, 1.2]    # Factor

        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_link_mass = True
        added_link_mass_range = [0.9, 1.1]

        randomize_motor_offset = True
        motor_offset_range = [-0.035, 0.035] # Offset to add to the motor angles
        
        randomize_joint_friction = True
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.01, 1.15]
        joint_1_friction_range = [0.01, 1.15]
        joint_2_friction_range = [0.01, 1.15]
        joint_3_friction_range = [0.01, 1.15]
        joint_4_friction_range = [0.5, 1.3]
        joint_5_friction_range = [0.5, 1.3]
        joint_6_friction_range = [0.01, 1.15]
        joint_7_friction_range = [0.01, 1.15]
        joint_8_friction_range = [0.01, 1.15]
        joint_9_friction_range = [0.5, 1.3]
        joint_10_friction_range = [0.5, 1.3]

        randomize_joint_damping = True
        randomize_joint_damping_each_joint = False
        joint_damping_range = [0.3, 1.5]
        joint_1_damping_range = [0.3, 1.5]
        joint_2_damping_range = [0.3, 1.5]
        joint_3_damping_range = [0.3, 1.5]
        joint_4_damping_range = [0.9, 1.5]
        joint_5_damping_range = [0.9, 1.5]
        joint_6_damping_range = [0.3, 1.5]
        joint_7_damping_range = [0.3, 1.5]
        joint_8_damping_range = [0.3, 1.5]
        joint_9_damping_range = [0.9, 1.5]
        joint_10_damping_range = [0.9, 1.5]

        randomize_joint_armature = True
        randomize_joint_armature_each_joint = False
        joint_armature_range = [0.0001, 0.05]     # Factor
        joint_1_armature_range = [0.0001, 0.05]
        joint_2_armature_range = [0.0001, 0.05]
        joint_3_armature_range = [0.0001, 0.05]
        joint_4_armature_range = [0.0001, 0.05]
        joint_5_armature_range = [0.0001, 0.05]
        joint_6_armature_range = [0.0001, 0.05]
        joint_7_armature_range = [0.0001, 0.05]
        joint_8_armature_range = [0.0001, 0.05]
        joint_9_armature_range = [0.0001, 0.05]
        joint_10_armature_range = [0.0001, 0.05]

        add_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [5, 40]
        
        add_dof_lag = True
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 40]
        
        add_dof_pos_vel_lag = False
        randomize_dof_pos_lag_timesteps = False
        randomize_dof_pos_lag_timesteps_perstep = False
        dof_pos_lag_timesteps_range = [7, 25]
        randomize_dof_vel_lag_timesteps = False
        randomize_dof_vel_lag_timesteps_perstep = False
        dof_vel_lag_timesteps_range = [7, 25]
        
        add_imu_lag = False
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [1, 10]
        
        randomize_coulomb_friction = True
        joint_coulomb_range = [0.1, 0.9]
        joint_viscous_range = [0.05, 0.1]
        
    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 25.  # time before command are changed[s]
        gait = ["walk_omnidirectional","stand","walk_omnidirectional"] # gait type during training
        # proportion during whole life time
        gait_time_range = {"walk_sagittal": [2,6],
                           "walk_lateral": [2,6],
                           "rotate": [2,3],
                           "stand": [2,3],
                           "walk_omnidirectional": [4,6]}

        heading_command = False  # if true: compute ang vel command from heading error
        stand_com_threshold = 0.05 # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand 改动log：pid 30420 - 20250113-1046
        sw_switch = True # use stand_com_threshold or not

        class ranges:
            lin_vel_x = [-0.4, 1.4] # min max [m/s] 
            lin_vel_y = [-0.4, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.7, 0.7]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        soft_dof_pos_limit = 0.98
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.90 #TODO:根据新机器人站立高度调整 log : PID  34406  20250113_1659
        foot_min_dist = 0.2
        foot_max_dist = 1.

        # final_swing_joint_pos = final_swing_joint_delta_pos + default_pos 和大腿方向弯曲一致根据default_joint_angles 设置  Process ID: 32414 -20250113_1354
        final_swing_joint_delta_pos =  [0.001, # 'Ll_root_joint'
                                       -0.001, #Ll_upper_joint
                                       -0.7, #Ll_knee_joint
                                       0.42, #Ll_lower_joint
                                       -0.14, # Lf_pitch_joint
                                       0.01, #Lf_row_joint
                                          
                                       -0.001, 
                                       0.001, 
                                       0.7, 
                                       -0.42, 
                                       -0.14, 
                                      0.01 ]
        target_feet_height = 0.03 #增加容忍，log: pid 31124-20250113_1126
        target_feet_height_max = 0.09 
        feet_to_ankle_distance = 0.036 #TODO:根据新机器人脚踝到地面距离调整 重新调整 log : PID  33366 20250113_1512
        cycle_time = 0.7
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(-error*sigma)
        tracking_sigma = 5 
        max_contact_force = 780  # 重量（39）*9.81 = 382.59 （35的重量应该是686.7 ~ 700，39的重量应该是765 ~ 720在测试，此处 780） log: pid 
        
        class scales:
            ref_joint_pos = 2.2 # 关节跟踪的重要性
            feet_clearance = 1. # 抬腿高度
            feet_contact_number = 2.0 # 支撑腿数量
            # gait
            feet_air_time = 1.2 # 摆动相持续时间
            foot_slip = -0.1 # 支撑相打滑惩罚
            feet_distance = 0.2 # 双脚间距
            knee_distance = 0.2 # 膝盖间距
            # contact 
            feet_contact_forces = -0.01 # 接触力惩罚
            # vel tracking
            tracking_lin_vel = 1.8
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.15
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 1.0
            orientation = 1. # 身体姿态维持
            feet_rotation = 0.3 # 足部旋转控制
            base_height = 0.2 # 躯干高度维持
            base_acc = 0.2 # 加速度平滑度
            # energy
            action_smoothness = -0.002 # 动作平滑度惩罚
            torques = -8e-9  # 关节力矩使用惩罚
            
            # 如果动作太保守，可以适当减小惩罚  log: id  31457-20250113_1134
            dof_vel = -2e-8  # 关节速度大小惩罚
            dof_acc = -1e-7  # 关节加速度惩罚
            
            collision = -1. # 碰撞惩罚
            stand_still = 2.5 # 静止状态奖励
            # limits
            dof_vel_limits = -1 # 速度限制惩罚
            dof_pos_limits = -10.  # 位置限制惩罚
            dof_torque_limits = -0.1 # 力矩限制惩罚

    class normalization:
        class obs_scales:
            lin_vel = 4.
            ang_vel = 3.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.


class KQLDHStandCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'DHOnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        state_estimator_hidden_dims=[256, 128, 64]
        
        #for long_history cnn only
        kernel_size=[6, 4]
        filter_size=[32, 16]
        stride_size=[3, 2]
        lh_output_dim= 64   #long history output dim
        in_channels = KQLDHStandCfg.env.frame_stack

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4
        if KQLDHStandCfg.terrain.measure_heights:
            lin_vel_idx = (KQLDHStandCfg.env.single_num_privileged_obs + KQLDHStandCfg.terrain.num_height) * (KQLDHStandCfg.env.c_frame_stack - 1) + KQLDHStandCfg.env.single_linvel_index
        else:
            lin_vel_idx = KQLDHStandCfg.env.single_num_privileged_obs * (KQLDHStandCfg.env.c_frame_stack - 1) + KQLDHStandCfg.env.single_linvel_index

    class runner:
        policy_class_name = 'ActorCriticDH'
        algorithm_class_name = 'DHPPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 20000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'kql_dh_stand'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
