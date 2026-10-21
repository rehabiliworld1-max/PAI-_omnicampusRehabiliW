"""MyoSuite (myoLegWalk-v0, 2023 environment) -> ROS 2 bridge helpers.

1. export_urdf(model, outdir)  : exact URDF + STL meshes generated from the compiled MuJoCo model
2. urdf_fk(...)                : forward kinematics of that URDF (used to verify it against MuJoCo)
3. Messages / RosStreamer      : sensor_msgs/JointState, tf2_msgs/TFMessage, visualization_msgs/MarkerArray
                                 (muscle paths coloured by activation), std_msgs/Float32MultiArray,
                                 published through rosbridge (roslibpy, no ROS install needed on this side)
"""
import os, json, time, struct
import numpy as np
import xml.etree.ElementTree as ET

HINGE, SLIDE, FREE = 3, 2, 0


# ---------------------------------------------------------------- rotations
def quat_to_mat(q):                      # MuJoCo order (w, x, y, z)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_rpy(R):                       # URDF: R = Rz(yaw) Ry(pitch) Rx(roll)
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(R[2, 1], R[2, 2]); yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1]); yaw = 0.0
    return roll, pitch, yaw


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, a):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def T(R=np.eye(3), p=(0, 0, 0)):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p; return M


def fmt(v):
    return " ".join(f"{x:.8g}" for x in v)


# ---------------------------------------------------------------- URDF export
def write_stl(path, verts, faces):
    tri = verts[faces]                                         # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    with open(path, "wb") as f:
        f.write(b"MyoSuite mesh exported for RViz".ljust(80, b" ")); f.write(struct.pack("<I", len(faces)))
        rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"] = n; rec["v"] = tri; f.write(rec.tobytes())


def export_urdf(m, outdir, mesh_uri_prefix, robot="myoleg"):
    """Write <outdir>/myoleg.urdf and <outdir>/meshes/*.stl from a compiled mujoco_py model.
    mesh_uri_prefix: how the ROS side sees <outdir>/meshes, e.g. 'file:///work/data/ros/meshes'.
    The free joint of 'root' is NOT in the URDF: publish it as TF world -> root."""
    os.makedirs(os.path.join(outdir, "meshes"), exist_ok=True)
    name = lambda b: m.body_id2name(b)
    rob = ET.Element("robot", name=robot)
    ET.SubElement(rob, "material", name="bone").append(ET.Element("color", rgba="0.92 0.89 0.80 1"))
    joint_names = []
    exported = set()

    def add_link(lname, body=None):
        L = ET.SubElement(rob, "link", name=lname)
        if body is None:
            return
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != body or m.geom_type[g] != 7:      # mesh geoms only (bones)
                continue
            mid = m.geom_dataid[g]; mname = m.mesh_id2name(mid) or f"mesh{mid}"
            if mid not in exported:
                va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]; fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
                write_stl(os.path.join(outdir, "meshes", f"{mname}.stl"), m.mesh_vert[va:va + vn].astype(np.float32), m.mesh_face[fa:fa + fn])
                exported.add(mid)
            V = ET.SubElement(L, "visual")
            ET.SubElement(V, "origin", xyz=fmt(m.geom_pos[g]), rpy=fmt(mat_to_rpy(quat_to_mat(m.geom_quat[g]))))
            Gm = ET.SubElement(V, "geometry"); ET.SubElement(Gm, "mesh", filename=f"{mesh_uri_prefix}/{mname}.stl")
            ET.SubElement(V, "material", name="bone")

    for b in range(1, m.nbody):
        bname = name(b); par = m.body_parentid[b]
        if par == 0:                                                # body attached to the world by the free joint
            add_link(bname, b); continue
        jids = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] in (HINGE, SLIDE)]
        R0 = quat_to_mat(m.body_quat[b]); p0 = m.body_pos[b]
        parent = name(par)
        if not jids:
            add_link(bname, b)
            J = ET.SubElement(rob, "joint", name=f"{parent}__to__{bname}", type="fixed")
            ET.SubElement(J, "parent", link=parent); ET.SubElement(J, "child", link=bname)
            ET.SubElement(J, "origin", xyz=fmt(p0), rpy=fmt(mat_to_rpy(R0)))
            continue
        for k, j in enumerate(jids):
            child = bname if k == len(jids) - 1 else f"{bname}__{k}"
            add_link(child, b if child == bname else None)
            jp = m.jnt_pos[j]
            if np.linalg.norm(jp) > 1e-12:
                raise NotImplementedError("joint with non-zero pos")   # not present in myoLeg
            jn = m.joint_id2name(j); joint_names.append(jn)
            J = ET.SubElement(rob, "joint", name=jn, type="revolute" if m.jnt_type[j] == HINGE else "prismatic")
            ET.SubElement(J, "parent", link=parent); ET.SubElement(J, "child", link=child)
            if k == 0:
                ET.SubElement(J, "origin", xyz=fmt(p0), rpy=fmt(mat_to_rpy(R0)))
            else:
                ET.SubElement(J, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(J, "axis", xyz=fmt(m.jnt_axis[j] / np.linalg.norm(m.jnt_axis[j])))
            lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-3.1416, 3.1416))
            ET.SubElement(J, "limit", lower=f"{lo:.6g}", upper=f"{hi:.6g}", effort="1000", velocity="100")
            parent = child
    ET.indent(rob) if hasattr(ET, "indent") else None
    path = os.path.join(outdir, f"{robot}.urdf")
    ET.ElementTree(rob).write(path, xml_declaration=True, encoding="utf-8")
    return path, joint_names


def urdf_fk(urdf_path, q, root_T):
    """Link world transforms from the URDF. q: {joint_name: value}; root_T: 4x4 world->root."""
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for J in root.findall("joint"):
        o = J.find("origin"); xyz = [float(v) for v in o.get("xyz").split()]; rpy = [float(v) for v in o.get("rpy").split()]
        ax = J.find("axis"); axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [1, 0, 0]
        joints[J.find("child").get("link")] = (J.find("parent").get("link"), J.get("name"), J.get("type"), xyz, rpy, axis)
    out = {}

    def get(link):
        if link in out:
            return out[link]
        if link not in joints:
            out[link] = root_T; return root_T
        par, jn, jt, xyz, rpy, axis = joints[link]
        M = get(par) @ T(rpy_to_mat(*rpy), xyz)
        if jt == "revolute":
            M = M @ T(axis_angle(axis, q.get(jn, 0.0)))
        elif jt == "prismatic":
            M = M @ T(p=np.asarray(axis) / np.linalg.norm(axis) * q.get(jn, 0.0))
        out[link] = M; return M

    for L in root.findall("link"):
        get(L.get("name"))
    return out


# ---------------------------------------------------------------- messages
def stamp(t=None):
    t = time.time() if t is None else t
    return {"sec": int(t), "nanosec": int((t - int(t)) * 1e9)}


def act_color(a):
    """activation 0 -> blue, 0.5 -> purple, 1 -> red"""
    a = float(np.clip(a, 0, 1))
    return {"r": 0.15 + 0.85 * a, "g": 0.25 * (1 - a), "b": 0.95 * (1 - a) + 0.1 * a, "a": 0.9}


class Messages:
    """Builds rosbridge (JSON) messages from the current MuJoCo state."""

    def __init__(self, m, joint_names):
        self.m = m
        self.joint_names = joint_names
        self.qadr = [m.get_joint_qpos_addr(j) for j in joint_names]
        self.dadr = [m.jnt_dofadr[m.joint_name2id(j)] for j in joint_names]
        self.muscles = [m.actuator_id2name(i) for i in range(m.nu)]
        self.tendon = [int(m.actuator_trnid[i][0]) for i in range(m.nu)]
        self.root_q = m.get_joint_qpos_addr(m.joint_id2name(0)) if m.joint_id2name(0) else (0, 7)
        self.root_name = m.body_id2name(m.jnt_bodyid[0])

    def joint_state(self, d, t):
        return {"header": {"stamp": stamp(t), "frame_id": ""}, "name": self.joint_names,
                "position": [float(d.qpos[a]) for a in self.qadr], "velocity": [float(d.qvel[a]) for a in self.dadr],
                "effort": [float(d.qfrc_actuator[a]) for a in self.dadr]}        # muscle torque about each joint [N m or N]

    def root_tf(self, d, t):
        q = d.qpos[0:7]                                                     # free joint: x y z, w x y z
        return {"transforms": [{"header": {"stamp": stamp(t), "frame_id": "world"}, "child_frame_id": self.root_name,
                                "transform": {"translation": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
                                              "rotation": {"x": float(q[4]), "y": float(q[5]), "z": float(q[6]), "w": float(q[3])}}}]}

    def muscle_paths(self, d):
        out = []
        for i, ti in enumerate(self.tendon):
            adr, num = d.ten_wrapadr[ti], d.ten_wrapnum[ti]
            out.append(np.array([d.wrap_xpos[k] for k in range(adr, adr + num) if d.wrap_obj[k] != -2]))   # -2 = pulley
        return out

    def muscle_markers(self, d, t, width=0.008):
        ms = []
        for i, P in enumerate(self.muscle_paths(d)):
            ms.append({"header": {"stamp": stamp(t), "frame_id": "world"}, "ns": "muscles", "id": i, "type": 4, "action": 0,
                       "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
                       "scale": {"x": width, "y": 0.0, "z": 0.0}, "color": act_color(d.act[i]),
                       "lifetime": {"sec": 0, "nanosec": 0}, "frame_locked": False,
                       "points": [{"x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "z": round(float(p[2]), 4)} for p in P]})
        return {"markers": ms}

    @staticmethod
    def array(values):
        return {"layout": {"dim": [], "data_offset": 0}, "data": [round(float(v), 4) for v in values]}


TOPICS = {  # topic: ROS 2 type
    "/joint_states": "sensor_msgs/msg/JointState",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/muscles": "visualization_msgs/msg/MarkerArray",
    "/muscle_activation": "std_msgs/msg/Float32MultiArray",
    "/muscle_force": "std_msgs/msg/Float32MultiArray",
    "/foot_contact": "std_msgs/msg/Float32MultiArray",
}


class RosStreamer:
    """Publishes the MuJoCo state to ROS 2 through rosbridge (websocket, port 9090)."""

    def __init__(self, host, port=9090, timeout=10):
        import roslibpy
        self.roslibpy = roslibpy
        self.ros = roslibpy.Ros(host=host, port=port)
        self.ros.run(timeout=timeout)
        self.pub = {}
        for tp, ty in TOPICS.items():
            p = roslibpy.Topic(self.ros, tp, ty, queue_size=1); p.advertise(); self.pub[tp] = p

    def publish(self, msgs, d, t):
        P, M = self.pub, self.roslibpy.Message
        P["/tf"].publish(M(msgs.root_tf(d, t)))
        P["/joint_states"].publish(M(msgs.joint_state(d, t)))
        P["/muscles"].publish(M(msgs.muscle_markers(d, t)))
        P["/muscle_activation"].publish(M(Messages.array(d.act)))
        P["/muscle_force"].publish(M(Messages.array(-d.actuator_force)))          # tension [N], positive
        P["/foot_contact"].publish(M(Messages.array(d.sensordata[:4])))            # r_foot r_toes l_foot l_toes [N]

    def close(self):
        for p in self.pub.values():
            try:
                p.unadvertise()
            except Exception:
                pass
        self.ros.close()          # terminate() would stop the twisted reactor for good (ReactorNotRestartable on re-run)


# ---------------------------------------------------------------- files for the ROS 2 container
RVIZ = """Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Enabled: true
      Reference Frame: world
      Plane: XY
      Plane Cell Count: 40
      Cell Size: 0.5
    - Class: rviz_default_plugins/RobotModel
      Name: Bones
      Enabled: true
      Description Source: Topic
      Description Topic:
        Value: /robot_description
      Alpha: 1
    - Class: rviz_default_plugins/MarkerArray
      Name: Muscles
      Enabled: true
      Topic:
        Value: /muscles
  Global Options:
    Fixed Frame: world
    Background Color: 48; 48; 48
    Frame Rate: 30
  Tools:
    - Class: rviz_default_plugins/MoveCamera
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Name: Current View
      Target Frame: pelvis
      Distance: 3.0
      Pitch: 0.25
      Yaw: 0.0
      Focal Point:
        X: 0
        Y: 0
        Z: 0
Window Geometry:
  Width: 1400
  Height: 900
"""

DOCKERFILE_ROS = """FROM osrf/ros:humble-desktop
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
        ros-humble-rosbridge-server ros-humble-robot-state-publisher \\
    && rm -rf /var/lib/apt/lists/*
"""

START_ROS = """#!/bin/bash
# Runs inside the ROS 2 container: rosbridge (websocket 9090) + robot_state_publisher + RViz
source /opt/ros/humble/setup.bash
D=/work/data/ros
ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=9090 > /tmp/rosbridge.log 2>&1 &
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(cat $D/myoleg.urdf)" > /tmp/rsp.log 2>&1 &
sleep 3
echo "rosbridge: ws://<this container>:9090   (logs: /tmp/rosbridge.log, /tmp/rsp.log)"
if [ "${NO_RVIZ:-0}" = "1" ]; then wait; else rviz2 -d $D/myoleg.rviz; fi
"""


def write_ros_files(outdir):
    for fn, txt in (("myoleg.rviz", RVIZ), ("Dockerfile.ros", DOCKERFILE_ROS), ("start_ros.sh", START_ROS)):
        with open(os.path.join(outdir, fn), "w", newline="\n") as f:
            f.write(txt)
