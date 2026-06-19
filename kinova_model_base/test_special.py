import mujoco
import mujoco_viewer
import numpy as np

model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
viewer.cam.distance = 3.0
viewer.cam.azimuth = 180.0#90
viewer.cam.elevation = -45
viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

obstacle_marker_name = "sphere1"
obstacle_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obstacle_marker_name)
model.body_pos[obstacle_marker_id] = [0.69554959, 0.02832093, 0.23395694+1.2]

goal_marker_name = "target"
goal_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, goal_marker_name)
model.body_pos[goal_marker_id] = [0.36115212, 0.7153158,  0.40791408+1.2]
qpos = data.qpos
q_start = [1.0700, 0.6577, -0.0731, +1.7501, 0.0183, 0.4865, -0.5] # [0.27966399, -0.4592131,   0.1065318+1.2]
qpos[0:7] = q_start
#print(data.qpos)
q_goal = [-1.0700, 0.6577, -0.0731, +1.0501, 0.0183, 0.4865, -0.5]
Kp = 0.7
i=0
frame_skip = 5
while data.time < 10:
    #data.ctrl[0]=-0.5
    #data.ctrl[1]=0.0
    #data.ctrl[2]=0.1
    #data.ctrl[3]=0.5
    #data.ctrl[4]=0.0
    #data.ctrl[5]=0.2
    #data.ctrl[6]=0.0
    #data.ctrl[7]=-0.2
    if i%5==0:
        for i in range(7):
            data.ctrl[i] = Kp*(q_goal[i] - data.qpos[i])
    
    r_pos = [data.joint('joint_1').qpos[0], data.joint('joint_2').qpos[0], data.joint('joint_3').qpos[0],
    data.joint('joint_4').qpos[0], data.joint('joint_5').qpos[0], data.joint('joint_6').qpos[0], data.joint('joint_7').qpos[0]]
    #print('j_pose:',' '.join(['%.4f'%x for x in r_pos]))
    #print(data.time)
    #x = (mjData.time, mjData.qpos, mjData.qvel, mjData.act)
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break
    i = i+1
## ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'left_coupler_joint', 'left_driver_joint', 'left_follower_joint', 'left_spring_link_joint', 'right_coupler_joint', 'right_driver_joint', 'right_follower_joint', 'right_spring_link_joint']
