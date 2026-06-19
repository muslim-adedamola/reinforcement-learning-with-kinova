import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)

while data.time < 10:
    #data.ctrl[0]=-0.5
    data.ctrl[1]=0.5
    #data.ctrl[2]=0.1
    #data.ctrl[3]=0.5
    #data.ctrl[4]=0.0
    #data.ctrl[5]=0.2
    #data.ctrl[6]=0.0
    #data.ctrl[7]=-0.2
    
    r_pos = [data.joint('joint_1').qpos[0], data.joint('joint_2').qpos[0], data.joint('joint_3').qpos[0],
    data.joint('joint_4').qpos[0], data.joint('joint_5').qpos[0], data.joint('joint_6').qpos[0], data.joint('joint_7').qpos[0]]
    #print('j_pose:',' '.join(['%.4f'%x for x in r_pos]))
    print(data.time)
    #x = (mjData.time, mjData.qpos, mjData.qvel, mjData.act)
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break
## ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'left_coupler_joint', 'left_driver_joint', 'left_follower_joint', 'left_spring_link_joint', 'right_coupler_joint', 'right_driver_joint', 'right_follower_joint', 'right_spring_link_joint']
