"""
dependencies:

- https://github.com/20tab/bvh-python
```
$ pip install bvh
```

- https://github.com/Zuzu-Typ/PyGLM
```
$ pip install pyglm
```
"""
from __future__ import annotations
from typing import List, TypedDict, Tuple, Optional, Dict, NamedTuple
from typing_extensions import NotRequired
import argparse
import pathlib
import bvh
import ctypes
import base64

import glm
import json
from enum import Enum, auto


class HumanBones(Enum):
    # 6
    Hips = auto()
    Spine = auto()
    Chest = auto()
    UpperChest = auto()
    Neck = auto()
    Head = auto()
    # 8
    LeftUpperLeg = auto()
    LeftLowerLeg = auto()
    LeftFoot = auto()
    LeftToes = auto()
    RightUpperLeg = auto()
    RightLowerLeg = auto()
    RightFoot = auto()
    RightToes = auto()
    # 8
    LeftShoulder = auto()
    LeftUpperArm = auto()
    LeftLowerArm = auto()
    LeftHand = auto()
    RightShoulder = auto()
    RightUpperArm = auto()
    RightLowerArm = auto()
    RightHand = auto()


def guess_humanoid(name: str) -> Optional[HumanBones]:
    match (name.lower()):
        case "hips":
            return HumanBones.Hips
        case "spine":
            return HumanBones.Spine
        case "spine1":
            return HumanBones.Chest
        case "neck":
            return HumanBones.Neck
        case "head":
            return HumanBones.Head
        case "leftshoulder":
            return HumanBones.LeftShoulder
        case "leftarm":
            return HumanBones.LeftUpperArm
        case "leftforearm":
            return HumanBones.LeftLowerArm
        case "lefthand":
            return HumanBones.LeftHand
        case "rightshoulder":
            return HumanBones.RightShoulder
        case "rightarm":
            return HumanBones.RightUpperArm
        case "rightforearm":
            return HumanBones.RightLowerArm
        case "righthand":
            return HumanBones.RightHand
        case "leftupleg":
            return HumanBones.LeftUpperLeg
        case "leftleg":
            return HumanBones.LeftLowerLeg
        case "leftfoot":
            return HumanBones.LeftFoot
        case "lefttoebase":
            return HumanBones.LeftToes
        case "rightupleg":
            return HumanBones.RightUpperLeg
        case "rightleg":
            return HumanBones.RightLowerLeg
        case "rightfoot":
            return HumanBones.RightFoot
        case "righttoebase":
            return HumanBones.RightToes
    print(f"## {name} not found ##")


class GltfAsset(TypedDict):
    version: str


class GltfScene(TypedDict):
    nodes: List[int]


class GltfNode(TypedDict):
    name: str
    translation: Tuple[float, float, float]
    children: NotRequired[List[int]]


class GltfBuffer(TypedDict):
    uri: str
    byteLength: int


class GltfBufferView(TypedDict):
    buffer: int
    byteOffset: int
    byteLength: int
    target: NotRequired[int]


class GltfAccessor(TypedDict):
    bufferView: int
    byteOffset: NotRequired[int]
    componentType: int
    count: int
    type: str
    max: NotRequired[List[float]]
    min: NotRequired[List[float]]


class GltfAnimationSampler(TypedDict):
    input: int
    interpolation: str
    output: int


class GltfAnimationTarget(TypedDict):
    node: int
    path: str


class GltfAnimationChannel(TypedDict):
    sampler: int
    target: GltfAnimationTarget


class GltfAnimation(TypedDict):
    samplers: List[GltfAnimationSampler]
    channels: List[GltfAnimationChannel]


class VrmAnimationHumanoidBone(TypedDict):
    node: int


class VrmAnimationHumanoidBones(TypedDict):
    hips: VrmAnimationHumanoidBone
    spine: VrmAnimationHumanoidBone
    chest: VrmAnimationHumanoidBone
    # upperChest: VrmAnimationHumanoidBone
    neck: VrmAnimationHumanoidBone
    head: VrmAnimationHumanoidBone
    leftShoulder: VrmAnimationHumanoidBone
    leftUpperArm: VrmAnimationHumanoidBone
    leftLowerArm: VrmAnimationHumanoidBone
    leftHand: VrmAnimationHumanoidBone
    rightShoulder: VrmAnimationHumanoidBone
    rightUpperArm: VrmAnimationHumanoidBone
    rightLowerArm: VrmAnimationHumanoidBone
    rightHand: VrmAnimationHumanoidBone
    leftUpperLeg: VrmAnimationHumanoidBone
    leftLowerLeg: VrmAnimationHumanoidBone
    leftFoot: VrmAnimationHumanoidBone
    leftToes: VrmAnimationHumanoidBone
    rightUpperLeg: VrmAnimationHumanoidBone
    rightLowerLeg: VrmAnimationHumanoidBone
    rightFoot: VrmAnimationHumanoidBone
    rightToes: VrmAnimationHumanoidBone
    # rightThumbDistal: VrmAnimationHumanoidBone
    # rightThumbProximal: VrmAnimationHumanoidBone
    # rightThumbMetacarpal: VrmAnimationHumanoidBone
    # rightIndexDistal: VrmAnimationHumanoidBone
    # rightIndexIntermediate: VrmAnimationHumanoidBone
    # rightIndexProximal: VrmAnimationHumanoidBone
    # rightMiddleDistal: VrmAnimationHumanoidBone
    # rightMiddleIntermediate: VrmAnimationHumanoidBone
    # rightMiddleProximal: VrmAnimationHumanoidBone
    # rightRingDistal: VrmAnimationHumanoidBone
    # rightRingIntermediate: VrmAnimationHumanoidBone
    # rightRingProximal: VrmAnimationHumanoidBone
    # rightLittleDistal: VrmAnimationHumanoidBone
    # rightLittleIntermediate: VrmAnimationHumanoidBone
    # rightLittleProximal: VrmAnimationHumanoidBone
    # leftThumbDistal: VrmAnimationHumanoidBone
    # leftThumbProximal: VrmAnimationHumanoidBone
    # leftThumbMetacarpal: VrmAnimationHumanoidBone
    # leftIndexDistal: VrmAnimationHumanoidBone
    # leftIndexIntermediate: VrmAnimationHumanoidBone
    # leftIndexProximal: VrmAnimationHumanoidBone
    # leftMiddleDistal: VrmAnimationHumanoidBone
    # leftMiddleIntermediate: VrmAnimationHumanoidBone
    # leftMiddleProximal: VrmAnimationHumanoidBone
    # leftRingDistal: VrmAnimationHumanoidBone
    # leftRingIntermediate: VrmAnimationHumanoidBone
    # leftRingProximal: VrmAnimationHumanoidBone
    # leftLittleDistal: VrmAnimationHumanoidBone
    # leftLittleIntermediate: VrmAnimationHumanoidBone
    # leftLittleProximal: VrmAnimationHumanoidBone


# class VrmAnimationHumanoid(TypedDict):
#     humanBones: object


class VrmAnimation(TypedDict):
    specVersion: str
    humanoid: VrmAnimationHumanoidBones


class GltfExtensions(TypedDict):
    VRMC_vrm_animation: NotRequired[VrmAnimation]


class Gltf(TypedDict):
    asset: GltfAsset
    buffers: List[GltfBuffer]
    bufferViews: List[GltfBufferView]
    accessors: List[GltfAccessor]
    #
    scene: int
    scenes: List[GltfScene]
    nodes: List[GltfNode]
    animations: List[GltfAnimation]
    #
    extensionsUsed: NotRequired[List[str]]
    extensions: NotRequired[GltfExtensions]


def print_bvh(node, indent=""):
    print(f"{indent}{node}")
    for child in node.children:
        print_bvh(child, indent + "  ")


def print_gltf(nodes, node_index, indent=""):
    node = nodes[node_index]
    print(f"{indent}{node['name']}: {node['translation']}")
    for child_index in node.get("children", []):
        print_gltf(nodes, child_index, indent + "  ")


class BvhChannelInfo(NamedTuple):
    gltf_node_index: int
    channels: List[str]


class BvhBuilder:
    def __init__(self, gltf_nodes: List[GltfNode]) -> None:
        self.gltf_nodes = gltf_nodes
        self.bvh_channels: List[BvhChannelInfo] = []

    def build_hierarchy(
        self,
        gltf_parent: GltfNode,
        bvh_node: bvh.BvhNode,
        scale: float,
    ):
        gltf_node_index = len(self.gltf_nodes)
        offset = bvh_node["OFFSET"]
        assert offset
        gltf_node: GltfNode = {
            "name": bvh_node.name,
            "translation": (
                float(offset[0]) * scale,
                float(offset[1]) * scale,
                float(offset[2]) * scale,
            ),
        }
        gltf_child_index = len(self.gltf_nodes)
        if "children" not in gltf_parent:
            gltf_parent["children"] = []
        self.gltf_nodes.append(gltf_node)
        gltf_parent["children"].append(gltf_child_index)

        for channels in bvh_node.filter("CHANNELS"):
            self.bvh_channels.append(
                BvhChannelInfo(gltf_node_index, channels.value[2:])
            )

        for child in bvh_node.filter("JOINT"):
            self.build_hierarchy(gltf_node, child, scale)
        for child in bvh_node.filter("End"):
            self.build_hierarchy(gltf_node, child, scale)


class GetBB:
    def __init__(self, gltf: Gltf) -> None:
        self.gltf = gltf
        self.min = glm.vec3(float("inf"), float("inf"), float("inf"))
        self.max = glm.vec3(-float("inf"), -float("inf"), -float("inf"))
        self.world_pos: Dict[int, glm.vec3] = {}

    def print(self, hips_index: int):
        print(
            f"""min: {self.min}
max: {self.max}
size: {self.max-self.min}
hips: {self.world_pos[hips_index]}
"""
        )

    def traverse(self, parent=glm.vec3(0, 0, 0), node_index=0, indent=""):
        node = self.gltf["nodes"][node_index]
        offset = node["translation"]
        pos = parent + glm.vec3(offset[0], offset[1], offset[2])
        self.world_pos[node_index] = pos
        self.min = glm.vec3(
            min(self.min.x, pos.x), min(self.min.y, pos.y), min(self.min.z, pos.z)
        )
        self.max = glm.vec3(
            max(self.max.x, pos.x), max(self.max.y, pos.y), max(self.max.z, pos.z)
        )
        # print(f'{indent}{node["name"]}: {pos}')
        for child_index in node.get("children", []):
            self.traverse(pos, child_index, indent + "  ")


class BinWriter:
    def __init__(self) -> None:
        self.bytes = bytearray()
        self.views: List[GltfBufferView] = []
        self.accessors: List[GltfAccessor] = []

    def push_float_array(self, value_elements: int, data: bytes) -> int:
        offset = len(self.bytes)
        self.bytes += data
        view_index = len(self.views)
        self.views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(data),
            }
        )
        accessor_index = len(self.accessors)
        value_type = ""
        count: int = 0
        match (value_elements):
            case 1:
                value_type = "SCALAR"
                count = len(data) // 4
            case 3:
                value_type = "VEC3"
                count = len(data) // 12
            case 4:
                value_type = "VEC4"
                count = len(data) // 16
            case _:
                raise Exception()
        self.accessors.append(
            {
                "bufferView": view_index,
                "componentType": 5126,  # float
                "type": value_type,
                "count": count,
            }
        )
        return accessor_index

    def to_base64(self) -> GltfBuffer:
        encoded = base64.b64encode(self.bytes)
        debug = base64.b64decode(encoded)
        assert debug == self.bytes
        return {
            "uri": (b"data:application/octet-stream;base64," + encoded).decode("ascii"),
            "byteLength": len(self.bytes),
        }


class BvhAnimation:
    def __init__(self, bin: BinWriter, scale: float) -> None:
        self.bin = bin
        self.scale = scale
        self.gltf_animation: GltfAnimation = {
            "samplers": [],
            "channels": [],
        }

    def bvh_animation(self, mocap: bvh.Bvh, channels: List[BvhChannelInfo]):
        input = (ctypes.c_float * mocap.nframes)()
        for i in range(mocap.nframes):
            input[i] = i * mocap.frame_time

        offset = 0
        for node_channel in channels:

            match node_channel.channels:
                case [
                    "Xposition",
                    "Yposition",
                    "Zposition",
                    "Zrotation",
                    "Xrotation",
                    "Yrotation",
                ]:
                    self.translation(
                        mocap,
                        input,
                        node_channel.gltf_node_index,
                        offset,
                        offset + 1,
                        offset + 2,
                    )
                    self.rotation_zxy(
                        mocap,
                        input,
                        node_channel.gltf_node_index,
                        offset + 3,
                        offset + 4,
                        offset + 5,
                    )

                case [
                    "Zrotation",
                    "Xrotation",
                    "Yrotation",
                ]:
                    self.rotation_zxy(
                        mocap,
                        input,
                        node_channel.gltf_node_index,
                        offset,
                        offset + 1,
                        offset + 2,
                    )

                case _:
                    raise NotImplemented()

            offset += len(node_channel.channels)

    def translation(
        self, mocap: bvh.Bvh, input, node_index: int, x: int, y: int, z: int
    ):
        output = (ctypes.c_float * (mocap.nframes * 3))()
        for i, f in enumerate(mocap.frames):
            output[i * 3 + 0] = float(f[x]) * self.scale
            output[i * 3 + 1] = float(f[y]) * self.scale
            output[i * 3 + 2] = float(f[z]) * self.scale
        gltf_sampler: GltfAnimationSampler = {
            "input": self.bin.push_float_array(1, memoryview(input).tobytes()),
            "interpolation": "LINEAR",
            "output": self.bin.push_float_array(3, memoryview(output).tobytes()),
        }
        sampler_index = len(self.gltf_animation["samplers"])
        self.gltf_animation["samplers"].append(gltf_sampler)
        gltf_channel: GltfAnimationChannel = {
            "sampler": sampler_index,
            "target": {
                "node": node_index,
                "path": "translation",
            },
        }
        self.gltf_animation["channels"].append(gltf_channel)

    def rotation_zxy(
        self, mocap: bvh.Bvh, input, node_index: int, z: int, x: int, y: int
    ):
        output = (ctypes.c_float * (mocap.nframes * 4))()
        for i, f in enumerate(mocap.frames):
            # TODO: euler
            rz = glm.quat(glm.radians(glm.vec3(0, 0, float(f[z]))))
            rx = glm.quat(glm.radians(glm.vec3(float(f[x]), 0, 0)))
            ry = glm.quat(glm.radians(glm.vec3(0, float(f[y]), 0)))
            q = rz * rx * ry
            # q = ry * rx * rz
            output[i * 4 + 0] = q.x
            output[i * 4 + 1] = q.y
            output[i * 4 + 2] = q.z
            output[i * 4 + 3] = q.w
        gltf_sampler: GltfAnimationSampler = {
            "input": self.bin.push_float_array(1, memoryview(input).tobytes()),
            "interpolation": "LINEAR",
            "output": self.bin.push_float_array(4, memoryview(output).tobytes()),
        }
        sampler_index = len(self.gltf_animation["samplers"])
        self.gltf_animation["samplers"].append(gltf_sampler)
        gltf_channel: GltfAnimationChannel = {
            "sampler": sampler_index,
            "target": {
                "node": node_index,
                "path": "rotation",
            },
        }
        self.gltf_animation["channels"].append(gltf_channel)


def convert(src: pathlib.Path, dst: pathlib.Path, scale: float):
    # load bvh
    mocap = bvh.Bvh(src.read_text())
    bvh_root = next(mocap.root.filter("ROOT"))
    # print_bvh(bvh_root)

    # gltf
    root: GltfNode = {
        "name": "__ROOT__",
        "translation": (0, 0, 0),
    }
    nodes: List[GltfNode] = [root]
    gltf: Gltf = {
        "asset": {"version": "2.0"},
        "extensionsUsed": [],
        "extensions": {},
        "accessors": [],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "animations": [],
        "buffers": [],
        "bufferViews": [],
    }
    bvh_builder = BvhBuilder(nodes)
    bvh_builder.build_hierarchy(root, bvh_root, scale)
    print_gltf(gltf["nodes"], 0)

    # animation
    bin = BinWriter()
    animation = BvhAnimation(bin, scale)
    animation.bvh_animation(mocap, bvh_builder.bvh_channels)
    # animation = bvh_animation(mocap, bin, scale)
    gltf["animations"].append(animation.gltf_animation)
    gltf["buffers"].append(bin.to_base64())
    gltf["bufferViews"] += bin.views
    gltf["accessors"] += bin.accessors

    # humanoid
    map = {}
    for i, node in enumerate(gltf["nodes"]):
        bone = guess_humanoid(node["name"])
        if bone:
            map[bone] = i
            print(f'{node["name"]} => {bone}')
    gltf["extensionsUsed"].append("VRMC_vrm_animation")
    gltf["extensions"]["VRMC_vrm_animation"] = {
        "specVersion": "1.0-draft",
        "humanoid": {
            # "humanBones": {
            "hips": {"node": map[HumanBones.Hips]},
            "spine": {"node": map[HumanBones.Spine]},
            "chest": {"node": map[HumanBones.Chest]},
            "neck": {"node": map[HumanBones.Neck]},
            "head": {"node": map[HumanBones.Head]},
            "leftShoulder": {"node": map[HumanBones.LeftShoulder]},
            "leftUpperArm": {"node": map[HumanBones.LeftUpperArm]},
            "leftLowerArm": {"node": map[HumanBones.LeftLowerArm]},
            "leftHand": {"node": map[HumanBones.LeftHand]},
            "rightShoulder": {"node": map[HumanBones.RightShoulder]},
            "rightUpperArm": {"node": map[HumanBones.RightUpperArm]},
            "rightLowerArm": {"node": map[HumanBones.RightLowerArm]},
            "rightHand": {"node": map[HumanBones.RightHand]},
            "leftUpperLeg": {"node": map[HumanBones.LeftUpperLeg]},
            "leftLowerLeg": {"node": map[HumanBones.LeftLowerLeg]},
            "leftFoot": {"node": map[HumanBones.LeftFoot]},
            "leftToes": {"node": map[HumanBones.LeftToes]},
            "rightUpperLeg": {"node": map[HumanBones.RightUpperLeg]},
            "rightLowerLeg": {"node": map[HumanBones.RightLowerLeg]},
            "rightFoot": {"node": map[HumanBones.RightFoot]},
            "rightToes": {"node": map[HumanBones.RightToes]},
            # }
        },
    }

    bb = GetBB(gltf)
    bb.traverse()
    bb.print(map[HumanBones.Hips])
    # get_bb(gltf)
    dst.write_text(json.dumps(gltf), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bvh", help="src bvh path", type=pathlib.Path)
    parser.add_argument("vrm", help="dst vrm path", type=pathlib.Path)
    parser.add_argument("--scale", help="scaling factor", type=float, default=1.0)
    args = parser.parse_args()
    convert(args.bvh, args.vrm, args.scale)


if __name__ == "__main__":
    main()
