from __future__ import annotations
from typing import List, TypedDict, Tuple, Optional, Dict
from typing_extensions import NotRequired
import argparse
import pathlib
import bvh
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


class VrmAnimationHumanoid(TypedDict):
    humanBones: object


class VrmAnimation(TypedDict):
    specVersion: str
    humanoid: VrmAnimationHumanoid


class GltfExtensions(TypedDict):
    VRMC_vrm_animation: VrmAnimation


class Gltf(TypedDict):
    asset: GltfAsset
    scene: int
    scenes: List[GltfScene]
    nodes: List[GltfNode]
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


def build_hierarchy(
    gltf_nodes: List[GltfNode], gltf_parent: GltfNode, bvh_node: bvh.BvhNode
):
    offset = bvh_node["OFFSET"]
    assert(offset)
    gltf_node: GltfNode = {
        "name": bvh_node.name,
        "translation": (float(offset[0]), float(offset[1]), float(offset[2])),
    }
    gltf_child_index = len(gltf_nodes)
    if "children" not in gltf_parent:
        gltf_parent["children"] = []
    gltf_nodes.append(gltf_node)
    gltf_parent["children"].append(gltf_child_index)

    for child in bvh_node.filter("JOINT"):
        build_hierarchy(gltf_nodes, gltf_node, child)
    for child in bvh_node.filter("End"):
        build_hierarchy(gltf_nodes, gltf_node, child)


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


def convert(src: pathlib.Path, dst: pathlib.Path):
    root: GltfNode = {
        "name": "__ROOT__",
        "translation": (0, 0, 0),
    }
    nodes: List[GltfNode] = [root]
    gltf: Gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
    }

    mocap = bvh.Bvh(src.read_text())
    bvh_root = next(mocap.root.filter("ROOT"))
    build_hierarchy(nodes, root, bvh_root)
    # print_bvh(bvh_root)
    print_gltf(gltf["nodes"], 0)

    map = {}
    for i, node in enumerate(gltf["nodes"]):
        bone = guess_humanoid(node["name"])
        if bone:
            map[bone] = i
            print(f'{node["name"]} => {bone}')

    gltf["extensionsUsed"] = ["VRMC_vrm_animation"]
    gltf["extensions"] = {
        "VRMC_vrm_animation": {
            "specVersion": "1.0-draft",
            "humanoid": {
                "humanBones": {
                    "hips": map[HumanBones.Hips],
                    "spine": map[HumanBones.Spine],
                    "chest": map[HumanBones.Chest],
                    "neck": map[HumanBones.Neck],
                    "head": map[HumanBones.Head],
                    "leftShoulder": map[HumanBones.LeftShoulder],
                    "leftUpperArm": map[HumanBones.LeftUpperArm],
                    "leftLowerArm": map[HumanBones.LeftLowerArm],
                    "leftHand": map[HumanBones.LeftHand],
                    "rightShoulder": map[HumanBones.RightShoulder],
                    "rightUpperArm": map[HumanBones.RightUpperArm],
                    "rightLowerArm": map[HumanBones.RightLowerArm],
                    "rightHand": map[HumanBones.RightHand],
                    "leftUpperLeg": map[HumanBones.LeftUpperLeg],
                    "leftLowerLeg": map[HumanBones.LeftLowerLeg],
                    "leftFoot": map[HumanBones.LeftFoot],
                    "leftToes": map[HumanBones.LeftToes],
                    "rightUpperLeg": map[HumanBones.RightUpperLeg],
                    "rightLowerLeg": map[HumanBones.RightLowerLeg],
                    "rightFoot": map[HumanBones.RightFoot],
                    "rightToes": map[HumanBones.RightToes],
                }
            },
        }
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
    args = parser.parse_args()
    convert(args.bvh, args.vrm)


if __name__ == "__main__":
    main()
